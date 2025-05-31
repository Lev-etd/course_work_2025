from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)


class QwenModel:
    """
    Wrapper for Qwen 2.5 language model to be used in the RAG pipeline.
    """

    def __init__(
        self,
        # model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        model_name: str = "Qwen/Qwen3-8B",
        device: str = None,
        use_fast_tokenizer: bool = True,
        max_length: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_half_precision: bool = True,
        force_cpu: bool = False,
        quantization: str = "4bit",  # None, "8bit", or "4bit"
    ):
        """
        Initialize the Qwen model.

        Args:
            model_name: The name or path of the model to load
            device: The device to run the model on (None for auto-detection)
            use_fast_tokenizer: Whether to use the fast tokenizer
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p sampling parameter
            use_half_precision: Whether to use FP16 (half precision)
            force_cpu: Whether to force CPU usage regardless of CUDA
            quantization: Quantization level (None, "8bit", "4bit")
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.quantization = quantization

        # Auto-detect device if not specified
        if device is None:
            if force_cpu:
                self.device = "cpu"
            else:
                # Check if CUDA is available and working properly
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    try:
                        # Try a simple CUDA operation to verify it's working
                        _ = torch.zeros(1, device="cuda")
                        self.device = "cuda"
                    except Exception as e:
                        logger.warning(f"CUDA error detected, falling back to CPU: {e}")
                        self.device = "cpu"
                else:
                    self.device = "cpu"
        else:
            self.device = device

        logger.info(
            f"Loading Qwen model {model_name} on {self.device} "
            f"with {quantization if quantization else 'no'} quantization"
        )

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=use_fast_tokenizer, trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise

        # Determine quantization configuration
        quantization_config = None
        if quantization == "4bit":
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quantization == "8bit":
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Determine dtype based on device and settings
        dtype = None
        if self.device == "cuda" and use_half_precision and not quantization:
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Try to load the model
        try:
            # Load with quantization if specified
            if quantization_config is not None:
                logger.info(f"Loading model with {quantization} quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            # CPU without quantization
            elif self.device == "cpu":
                logger.info("Loading model on CPU without quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                # Move model to CPU explicitly
                self.model = self.model.to("cpu")
            # CUDA without quantization
            else:
                logger.info("Loading model on CUDA without quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                )
            logger.info("Qwen model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

            # Try with 4-bit quantization if initial load failed
            if quantization is None or quantization != "4bit":
                try:
                    logger.info("Trying to load with 4-bit quantization instead")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    logger.info("Successfully loaded model with 4-bit quantization")
                    self.quantization = "4bit"
                except Exception as quant_e:
                    logger.error(f"Failed with 4-bit quantization as well: {quant_e}")
                    # Last resort - try CPU
                    if not force_cpu:
                        try:
                            logger.info("Trying to load model on CPU as last resort")
                            self.device = "cpu"
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                torch_dtype=torch.float32,
                                trust_remote_code=True,
                            )
                            self.model = self.model.to("cpu")
                            logger.info("Successfully loaded model on CPU")
                        except Exception as cpu_e:
                            logger.error(f"Failed on CPU as well: {cpu_e}")
                            raise
                    else:
                        raise
            else:
                raise

    def generate(self, prompt: str, max_length: Optional[int] = None) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            max_length: Optional override for max generation length

        Returns:
            The generated text
        """
        max_len = max_length or self.max_length

        # Format prompt according to Qwen's chat template
        messages = [{"role": "user", "content": prompt}]

        # Use the model's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Manual formatting if template not available
            prompt = f"USER: {prompt}\nASSISTANT: "

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_len,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:", 1)[1].strip()

        return response

    def generate_with_context(
        self, question: str, context: str, max_length: Optional[int] = None
    ) -> str:
        """
        Generate text using a question and context.

        Args:
            question: The user's question
            context: The context information from the knowledge base
            max_length: Optional override for max generation length

        Returns:
            The generated answer
        """
        prompt = f"""
You are an expert in industrial equipment and cooling systems.
Use the following context to answer the question at the end. 
If you don't know the answer, say you don't know.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        return self.generate(prompt, max_length)
