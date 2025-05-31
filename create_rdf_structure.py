from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
import random
import uuid

# Define namespaces with proper URIs for Virtuoso
ISO15926 = Namespace("http://standards.iso.org/iso/15926/-4/")
PLANT = Namespace("http://example.org/plant/")
DATA = Namespace("http://example.org/data/")


def create_base_graph():
    g = Graph()
    # Bind namespaces - important for Virtuoso
    g.bind("iso15926", ISO15926)
    g.bind("plant", PLANT)
    g.bind("data", DATA)
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    return g


def add_class_hierarchy(g):
    # Add PropertyType class first
    property_ns = Namespace("http://standards.iso.org/iso/15926/-4/property/")
    g.bind("property", property_ns)

    g.add((ISO15926.PropertyType, RDF.type, OWL.Class))
    g.add(
        (
            ISO15926.PropertyType,
            RDFS.label,
            Literal("PropertyType", datatype=XSD.string),
        )
    )

    # Extended class hierarchy with more specific types and deeper levels
    classes = {
        "Pump": [
            "CentrifugalPump",
            "PositiveDisplacementPump",
            "InlinePump",
            "VerticalPump",
        ],
        "CentrifugalPump": [
            "SingleStagePump",
            "MultiStagePump",
            "AxialFlowPump",
            "MixedFlowPump",
            "SelfPrimingPump",
        ],
        # Deeper pump hierarchy - SingleStagePump subclasses
        "SingleStagePump": [
            "EndSuctionPump",
            "CirculatorPump",
            "OverhungImpellerPump",
        ],
        "EndSuctionPump": [
            "FrameMountedPump",
            "CloseCoupledPump",
        ],
        "OverhungImpellerPump": [
            "ChemicalProcessPump",
            "APIProcessPump",
        ],
        "APIProcessPump": [
            "OH1Pump",
            "OH2Pump",
            "OH3Pump",
        ],
        "CirculatorPump": [
            "WetRotorCirculator",
            "DryRotorCirculator",
        ],
        # Deeper pump hierarchy - MultiStagePump subclasses
        "MultiStagePump": [
            "HorizontalMultiStagePump",
            "VerticalMultiStagePump",
            "BarrelMultiStagePump",
        ],
        "HorizontalMultiStagePump": [
            "BetweenBearingMultiStagePump",
            "RingSelectionMultiStagePump",
        ],
        "VerticalMultiStagePump": [
            "CanTypeMultiStagePump",
            "LineShaftMultiStagePump",
            "SubmersibleMultiStagePump",
        ],
        "PositiveDisplacementPump": [
            "GearPump",
            "ScrewPump",
            "RotaryLobePump",
            "ProgressiveCavityPump",
            "DiaphragmPump",
            "PistonPump",
        ],
        # Deeper pump hierarchy - PositiveDisplacement subclasses
        "GearPump": [
            "ExternalGearPump",
            "InternalGearPump",
            "GerotorPump",
        ],
        "ScrewPump": [
            "SingleScrewPump",
            "TwinScrewPump",
            "TripleScrewPump",
        ],
        "PistonPump": [
            "RadialPistonPump",
            "AxialPistonPump",
            "PlungerPump",
        ],
        "Valve": [
            "ControlValve",
            "ShutoffValve",
            "CheckValve",
            "PressureReliefValve",
            "ThreeWayValve",
        ],
        "ControlValve": [
            "GlobeControlValve",
            "ButterflyControlValve",
            "BallControlValve",
            "DiaphragmControlValve",
            "EccentricPlugValve",
        ],
        # Deeper valve hierarchy - GlobeControlValve subclasses
        "GlobeControlValve": [
            "AngleBodyGlobeValve",
            "StraightBodyGlobeValve",
            "YPatternGlobeValve",
        ],
        "YPatternGlobeValve": [
            "StandardTrimValve",
            "NoiseReductionTrimValve",
        ],
        "NoiseReductionTrimValve": [
            "MultipathTrimValve",
            "LabyrinthTrimValve",
        ],
        # Deeper valve hierarchy - BallControlValve subclasses
        "BallControlValve": [
            "TrunnionMountedBallValve",
            "FloatingBallValve",
            "VNotchBallValve",
        ],
        "TrunnionMountedBallValve": [
            "DoubleTrunnionValve",
            "TripleTrunnionValve",
        ],
        "VNotchBallValve": [
            "ThirtyDegreeVNotchValve",
            "SixtyDegreeVNotchValve",
            "NinetyDegreeVNotchValve",
        ],
        # Deeper valve hierarchy - ButterflyControlValve subclasses
        "ButterflyControlValve": [
            "HighPerformanceButterflyValve",
            "ResilientsSeatedButterflyValve",
            "TripleOffsetButterflyValve",
        ],
        "ShutoffValve": [
            "GateValve",
            "BallValve",
            "ButterflyValve",
            "PlugValve",
            "NeedleValve",
        ],
        "HeatExchanger": [
            "ShellAndTubeExchanger",
            "PlateHeatExchanger",
            "BrazedPlateHeatExchanger",
            "CoolingTower",
        ],
        "Sensor": ["TemperatureSensor", "PressureSensor", "FlowSensor", "LevelSensor"],
        # Deeper sensor hierarchy - TemperatureSensor subclasses
        "TemperatureSensor": [
            "RTDSensor",
            "ThermocoupleType",
            "ThermistorSensor",
            "InfraredTemperatureSensor",
        ],
        "RTDSensor": [
            "PT100Sensor",
            "PT1000Sensor",
            "HighPrecisionRTD",
        ],
        "ThermocoupleType": [
            "TypeJThermocouple",
            "TypeKThermocouple",
            "TypeSThermocouple",
            "TypeNThermocouple",
        ],
        # Deeper sensor hierarchy - PressureSensor subclasses
        "PressureSensor": [
            "PiezoelectricSensor",
            "CapacitanceSensor",
            "StrainGaugeSensor",
        ],
        "StrainGaugeSensor": [
            "BondedStrainGauge",
            "UnbondedStrainGauge",
        ],
        # Flow sensor subclasses
        "FlowSensor": [
            "DifferentialPressureFlowmeter",
            "VolumetricFlowmeter",
            "VelocityFlowmeter",
            "MassFlowmeter",
        ],
        "Controller": [
            "PIDController",
            "TemperatureController",
            "PressureController",
            "FlowController",
        ],
        # PID Controller subclasses
        "PIDController": [
            "SingleLoopController",
            "MultiLoopController",
            "CascadeController",
        ],
        "Tank": ["StorageTank", "ExpansionTank", "BufferTank"],
        "Filter": ["StrainerFilter", "CartridgeFilter", "SandFilter"],
    }

    # Add classes with proper typing and relationships
    for parent, children in classes.items():
        parent_uri = ISO15926[parent]
        g.add((parent_uri, RDF.type, OWL.Class))
        g.add((parent_uri, RDFS.label, Literal(parent, datatype=XSD.string)))

        for child in children:
            child_uri = ISO15926[child]
            g.add((child_uri, RDF.type, OWL.Class))
            g.add((child_uri, RDFS.label, Literal(child, datatype=XSD.string)))
            g.add((child_uri, RDFS.subClassOf, parent_uri))

    # Add industry standards relationships
    g.add((ISO15926.APIProcessPump, ISO15926.conformsTo, ISO15926.API610Standard))
    g.add((ISO15926.API610Standard, RDF.type, ISO15926.Standard))
    g.add(
        (
            ISO15926.API610Standard,
            RDFS.label,
            Literal("API 610 Standard", datatype=XSD.string),
        )
    )

    g.add(
        (ISO15926.ChemicalProcessPump, ISO15926.conformsTo, ISO15926.ANSIASMEStandard)
    )
    g.add((ISO15926.ANSIASMEStandard, RDF.type, ISO15926.Standard))
    g.add(
        (
            ISO15926.ANSIASMEStandard,
            RDFS.label,
            Literal("ANSI/ASME B73.1 Standard", datatype=XSD.string),
        )
    )

    return g


def add_property_definitions(g):
    """Add property type definitions to the ontology"""
    property_ns = Namespace("http://standards.iso.org/iso/15926/-4/property/")

    # Basic property types
    property_types = [
        "flow_rate",
        "head",
        "efficiency",
        "power",
        "npsh_required",
        "pressure",
        "temperature",
        "cv_value",
        "rangeability",
        "leakage_class",
        "fail_position",
        "measured_value",
        "accuracy",
        "response_time",
        "kp",
        "ki",
        "kd",
        "control_mode",
        "heat_rejection",
        "approach_temp",
        "air_flow",
        "tower_type",
        # Extended pump properties
        "number_of_stages",
        "impeller_diameter",
        "suction_size",
        "discharge_size",
        "frame_size",
        "bearing_type",
        "coupling_type",
        "base_plate_material",
        "motor_enclosure",
        "motor_efficiency_class",
        "direct_mount_type",
        "max_operating_temperature",
        "connection_type",
        "rotor_can_material",
        "mechanical_seal_type",
        "motor_protection_class",
        "max_solids_size",
        "casing_material",
        "impeller_material",
        "ansi_flange_rating",
        "standard_dimension",
        "back_pullout_design",
        "api_standard",
        "api_material_class",
        "pressure_rating",
        "temperature_rating",
        "centerline_type",
        "shaft_position",
        "mounting_arrangement",
        "balancing_drum_type",
        "shaft_sealing_arrangement",
        "bearing_lubrication",
        "suction_position",
        "column_length",
        "bowl_assembly_material",
        "motor_cooling",
        "cable_entry_type",
        "motor_enclosure_class",
        "max_pressure",
        "viscosity_rating",
        "speed",
        "displacement",
        "gear_teeth_type",
        "gear_material",
        "tooth_count",
        "rotor_type",
        "idler_gear_material",
        "crescent_type",
        # Extended valve properties
        "stem_diameter",
        "flow_characteristic",
        "angle_degrees",
        "valve_body_material",
        "plug_material",
        "face_to_face_dimension",
        "y_angle",
        "extension_bonnet",
        "noise_reduction_rating",
        "max_allowable_velocity",
        "special_trim_material",
        "number_of_paths",
        "path_configuration",
        "disc_type",
        "seat_material",
        "torque_requirement",
        "fire_safe_rating",
        "metal_seat_hardness",
        "ball_type",
        "port_size",
        "v_notch_angle",
        "trunnion_material",
        "seat_injection_system",
        "mounting_style",
        # Extended sensor properties
        "sensor_type",
        "resistance_at_0c",
        "wire_configuration",
        "element_material",
        "iec_class",
        "thermocouple_type",
        "junction_type",
        "positive_leg_material",
        "negative_leg_material",
        "bridge_configuration",
        "excitation_voltage",
        # Extended controller properties
        "input_type",
        "output_type",
        "kp_primary",
        "ki_primary",
        "kd_primary",
        "kp_secondary",
        "ki_secondary",
        "kd_secondary",
    ]

    for prop_type in property_types:
        prop_uri = property_ns[prop_type]
        g.add((prop_uri, RDF.type, ISO15926["PropertyType"]))
        print(f"Added property type: {prop_type}")


def generate_equipment_instance(g, equipment_type, tag_prefix):
    """Generate equipment instance with properties for Virtuoso"""
    instance_id = str(uuid.uuid4())
    instance_uri = DATA[instance_id]

    property_ns = Namespace("http://standards.iso.org/iso/15926/-4/property/")
    has_property = ISO15926["hasProperty"]
    has_value = ISO15926["hasValue"]
    has_unit = ISO15926["hasUnit"]

    # Extended properties dictionary with more realistic values for various equipment types
    properties = {
        # Pump Types
        "CentrifugalPump": {
            "flow_rate": (lambda: random.uniform(100, 500), "m3/h", XSD.decimal),
            "head": (lambda: random.uniform(20, 100), "m", XSD.decimal),
            "efficiency": (lambda: random.uniform(0.7, 0.9), None, XSD.decimal),
            "power": (lambda: random.uniform(50, 200), "kW", XSD.decimal),
            "npsh_required": (lambda: random.uniform(2, 8), "m", XSD.decimal),
        },
        "SingleStagePump": {
            "flow_rate": (lambda: random.uniform(50, 300), "m3/h", XSD.decimal),
            "head": (lambda: random.uniform(15, 60), "m", XSD.decimal),
            "efficiency": (lambda: random.uniform(0.65, 0.85), None, XSD.decimal),
            "power": (lambda: random.uniform(30, 150), "kW", XSD.decimal),
            "npsh_required": (lambda: random.uniform(1.5, 6), "m", XSD.decimal),
            "impeller_diameter": (lambda: random.uniform(200, 400), "mm", XSD.decimal),
        },
        # Deep pump hierarchy - SingleStagePump subclasses
        "EndSuctionPump": {
            "flow_rate": (lambda: random.uniform(40, 280), "m3/h", XSD.decimal),
            "head": (lambda: random.uniform(15, 55), "m", XSD.decimal),
            "efficiency": (lambda: random.uniform(0.65, 0.83), None, XSD.decimal),
            "power": (lambda: random.uniform(25, 140), "kW", XSD.decimal),
            "npsh_required": (lambda: random.uniform(1.4, 5.5), "m", XSD.decimal),
            "impeller_diameter": (lambda: random.uniform(180, 380), "mm", XSD.decimal),
            "suction_size": (lambda: random.uniform(50, 200), "mm", XSD.decimal),
            "discharge_size": (lambda: random.uniform(40, 150), "mm", XSD.decimal),
        },
        "FrameMountedPump": {
            "frame_size": (
                lambda: random.choice(["Frame 1", "Frame 2", "Frame 3", "Frame 4"]),
                None,
                XSD.string,
            ),
            "bearing_type": (
                lambda: random.choice(["Ball", "Roller", "Angular Contact"]),
                None,
                XSD.string,
            ),
            "coupling_type": (
                lambda: random.choice(["Flexible", "Rigid", "Spacer"]),
                None,
                XSD.string,
            ),
            "base_plate_material": (
                lambda: random.choice(["Carbon Steel", "Stainless Steel", "Cast Iron"]),
                None,
                XSD.string,
            ),
        },
        "CloseCoupledPump": {
            "motor_enclosure": (
                lambda: random.choice(["TEFC", "ODP", "Explosion Proof"]),
                None,
                XSD.string,
            ),
            "motor_efficiency_class": (
                lambda: random.choice(["IE2", "IE3", "IE4", "NEMA Premium"]),
                None,
                XSD.string,
            ),
            "direct_mount_type": (
                lambda: random.choice(["C-Face", "D-Flange", "Adapter"]),
                None,
                XSD.string,
            ),
        },
        "CirculatorPump": {
            "flow_rate": (lambda: random.uniform(0.5, 50), "m3/h", XSD.decimal),
            "head": (lambda: random.uniform(2, 15), "m", XSD.decimal),
            "efficiency": (lambda: random.uniform(0.45, 0.75), None, XSD.decimal),
            "power": (lambda: random.uniform(0.05, 3), "kW", XSD.decimal),
            "max_operating_temperature": (
                lambda: random.uniform(80, 120),
                "°C",
                XSD.decimal,
            ),
            "connection_type": (
                lambda: random.choice(["Threaded", "Flanged", "Union"]),
                None,
                XSD.string,
            ),
        },
        "WetRotorCirculator": {
            "rotor_can_material": (
                lambda: random.choice(["Stainless Steel", "Bronze", "Composite"]),
                None,
                XSD.string,
            ),
            "bearing_type": (
                lambda: random.choice(["Carbon", "Ceramic", "Composite"]),
                None,
                XSD.string,
            ),
            "speed_settings": (
                lambda: random.choice(["Single Speed", "3-Speed", "Variable Speed"]),
                None,
                XSD.string,
            ),
        },
        "DryRotorCirculator": {
            "mechanical_seal_type": (
                lambda: random.choice(["Single", "Double", "Cartridge"]),
                None,
                XSD.string,
            ),
            "motor_protection_class": (
                lambda: random.choice(["IP44", "IP54", "IP55"]),
                None,
                XSD.string,
            ),
        },
        "OverhungImpellerPump": {
            "max_solids_size": (lambda: random.uniform(1, 10), "mm", XSD.decimal),
            "casing_material": (
                lambda: random.choice(
                    [
                        "Cast Iron",
                        "Ductile Iron",
                        "Carbon Steel",
                        "316 Stainless Steel",
                        "Duplex Stainless Steel",
                    ]
                ),
                None,
                XSD.string,
            ),
            "impeller_material": (
                lambda: random.choice(
                    [
                        "Cast Iron",
                        "Bronze",
                        "316 Stainless Steel",
                        "Duplex Stainless Steel",
                        "Nickel Aluminum Bronze",
                    ]
                ),
                None,
                XSD.string,
            ),
        },
        "ChemicalProcessPump": {
            "ansi_flange_rating": (
                lambda: random.choice(["150#", "300#", "600#"]),
                None,
                XSD.string,
            ),
            "standard_dimension": (
                lambda: random.choice(["ANSI B73.1", "ANSI B73.2", "ANSI B73.3"]),
                None,
                XSD.string,
            ),
            "back_pullout_design": (
                lambda: random.choice(["True", "False"]),
                None,
                XSD.string,
            ),
        },
        "APIProcessPump": {
            "api_standard": (lambda: "API 610", None, XSD.string),
            "api_material_class": (
                lambda: random.choice(
                    [
                        "S-1",
                        "S-3",
                        "S-4",
                        "S-5",
                        "S-6",
                        "S-8",
                        "C-6",
                        "A-7",
                        "A-8",
                        "D-1",
                        "D-2",
                    ]
                ),
                None,
                XSD.string,
            ),
            "pressure_rating": (lambda: random.uniform(10, 100), "bar", XSD.decimal),
            "temperature_rating": (lambda: random.uniform(120, 450), "°C", XSD.decimal),
        },
        "OH1Pump": {
            "centerline_type": (lambda: "Foot mounted", None, XSD.string),
            "shaft_position": (lambda: "Horizontal", None, XSD.string),
            "mounting_arrangement": (lambda: "End-Suction, Overhung", None, XSD.string),
        },
        "OH2Pump": {
            "centerline_type": (lambda: "Centerline mounted", None, XSD.string),
            "shaft_position": (lambda: "Horizontal", None, XSD.string),
            "mounting_arrangement": (lambda: "End-Suction, Overhung", None, XSD.string),
        },
        "OH3Pump": {
            "centerline_type": (lambda: "In-line mounted", None, XSD.string),
            "shaft_position": (lambda: "Vertical", None, XSD.string),
            "mounting_arrangement": (lambda: "Inline, Overhung", None, XSD.string),
        },
        "MultiStagePump": {
            "flow_rate": (lambda: random.uniform(20, 200), "m3/h", XSD.decimal),
            "head": (lambda: random.uniform(100, 400), "m", XSD.decimal),
            "efficiency": (lambda: random.uniform(0.7, 0.82), None, XSD.decimal),
            "power": (lambda: random.uniform(75, 250), "kW", XSD.decimal),
            "npsh_required": (lambda: random.uniform(3, 10), "m", XSD.decimal),
            "number_of_stages": (lambda: random.randint(2, 12), None, XSD.integer),
        },
        "HorizontalMultiStagePump": {
            "balancing_drum_type": (
                lambda: random.choice(["Standard", "Enhanced", "Double"]),
                None,
                XSD.string,
            ),
            "shaft_sealing_arrangement": (
                lambda: random.choice(
                    ["Single Mechanical", "Double Mechanical", "Cartridge"]
                ),
                None,
                XSD.string,
            ),
            "bearing_lubrication": (
                lambda: random.choice(
                    ["Grease", "Oil Mist", "Forced Oil", "Self-Lubricating"]
                ),
                None,
                XSD.string,
            ),
        },
        "VerticalMultiStagePump": {
            "suction_position": (
                lambda: random.choice(["Top", "Bottom", "Side"]),
                None,
                XSD.string,
            ),
            "column_length": (lambda: random.uniform(1, 10), "m", XSD.decimal),
            "bowl_assembly_material": (
                lambda: random.choice(
                    ["Cast Iron", "Carbon Steel", "316 SS", "Duplex SS"]
                ),
                None,
                XSD.string,
            ),
        },
        "SubmersibleMultiStagePump": {
            "motor_cooling": (
                lambda: random.choice(["Water Filled", "Oil Filled", "Forced Cooling"]),
                None,
                XSD.string,
            ),
            "cable_entry_type": (
                lambda: random.choice(
                    ["Resin Sealed", "Mechanical Seal", "Cable Gland"]
                ),
                None,
                XSD.string,
            ),
            "motor_enclosure_class": (
                lambda: random.choice(["IP68", "IP67", "IP66"]),
                None,
                XSD.string,
            ),
        },
        "GearPump": {
            "flow_rate": (lambda: random.uniform(1, 50), "m3/h", XSD.decimal),
            "max_pressure": (lambda: random.uniform(5, 25), "bar", XSD.decimal),
            "viscosity_rating": (lambda: random.uniform(20, 1000), "cSt", XSD.decimal),
            "speed": (lambda: random.uniform(100, 1200), "rpm", XSD.decimal),
            "displacement": (lambda: random.uniform(10, 200), "cc/rev", XSD.decimal),
        },
        "ExternalGearPump": {
            "gear_teeth_type": (
                lambda: random.choice(["Spur", "Helical", "Herringbone"]),
                None,
                XSD.string,
            ),
            "gear_material": (
                lambda: random.choice(["Carbon Steel", "Stainless Steel", "Bronze"]),
                None,
                XSD.string,
            ),
            "tooth_count": (lambda: random.randint(6, 14), None, XSD.integer),
        },
        "InternalGearPump": {
            "rotor_type": (
                lambda: random.choice(["Straight Tooth", "Helical", "Curved Tooth"]),
                None,
                XSD.string,
            ),
            "idler_gear_material": (
                lambda: random.choice(["PEEK", "Steel", "Bronze", "Stainless Steel"]),
                None,
                XSD.string,
            ),
            "crescent_type": (
                lambda: random.choice(["Fixed", "Removable", "Integrated"]),
                None,
                XSD.string,
            ),
        },
        # Valve Types
        "ControlValve": {
            "cv_value": (lambda: random.uniform(10, 100), None, XSD.decimal),
            "rangeability": (lambda: random.uniform(20, 100), None, XSD.decimal),
            "leakage_class": (
                lambda: random.choice(["II", "III", "IV", "V", "VI"]),
                None,
                XSD.string,
            ),
            "fail_position": (
                lambda: random.choice(["Open", "Closed", "Last"]),
                None,
                XSD.string,
            ),
        },
        "GlobeControlValve": {
            "cv_value": (lambda: random.uniform(5, 80), None, XSD.decimal),
            "rangeability": (lambda: random.uniform(30, 100), None, XSD.decimal),
            "leakage_class": (
                lambda: random.choice(["IV", "V", "VI"]),
                None,
                XSD.string,
            ),
            "fail_position": (
                lambda: random.choice(["Open", "Closed"]),
                None,
                XSD.string,
            ),
            "stem_diameter": (lambda: random.uniform(10, 30), "mm", XSD.decimal),
            "flow_characteristic": (
                lambda: random.choice(["Linear", "Equal%", "Quick Open"]),
                None,
                XSD.string,
            ),
        },
        "AngleBodyGlobeValve": {
            "angle_degrees": (lambda: 90.0, "degrees", XSD.decimal),
            "valve_body_material": (
                lambda: random.choice(
                    ["Cast Iron", "Carbon Steel", "Stainless Steel", "Bronze", "Monel"]
                ),
                None,
                XSD.string,
            ),
            "plug_material": (
                lambda: random.choice(["316 SS", "17-4PH", "Stellite", "Ceramic"]),
                None,
                XSD.string,
            ),
        },
        "StraightBodyGlobeValve": {
            "valve_body_material": (
                lambda: random.choice(
                    ["Cast Iron", "Carbon Steel", "Stainless Steel", "Bronze", "Monel"]
                ),
                None,
                XSD.string,
            ),
            "plug_material": (
                lambda: random.choice(["316 SS", "17-4PH", "Stellite", "Ceramic"]),
                None,
                XSD.string,
            ),
            "face_to_face_dimension": (
                lambda: random.choice(["ANSI/ISA 75.08", "EN 558-1", "API 6D"]),
                None,
                XSD.string,
            ),
        },
        "YPatternGlobeValve": {
            "valve_body_material": (
                lambda: random.choice(
                    ["Cast Iron", "Carbon Steel", "Stainless Steel", "Bronze", "Monel"]
                ),
                None,
                XSD.string,
            ),
            "y_angle": (lambda: random.uniform(30, 60), "degrees", XSD.decimal),
            "extension_bonnet": (
                lambda: random.choice(["Yes", "No"]),
                None,
                XSD.string,
            ),
        },
        "NoiseReductionTrimValve": {
            "noise_reduction_rating": (
                lambda: random.uniform(15, 30),
                "dBA",
                XSD.decimal,
            ),
            "max_allowable_velocity": (
                lambda: random.uniform(50, 200),
                "m/s",
                XSD.decimal,
            ),
            "special_trim_material": (
                lambda: random.choice(["Hardened SS", "Tungsten Carbide", "Ceramic"]),
                None,
                XSD.string,
            ),
        },
        "MultipathTrimValve": {
            "number_of_paths": (lambda: random.randint(3, 6), None, XSD.integer),
            "path_configuration": (
                lambda: random.choice(["Series", "Parallel", "Hybrid"]),
                None,
                XSD.string,
            ),
        },
        "ButterflyControlValve": {
            "cv_value": (lambda: random.uniform(20, 1200), None, XSD.decimal),
            "rangeability": (lambda: random.uniform(15, 50), None, XSD.decimal),
            "leakage_class": (
                lambda: random.choice(["II", "III", "IV"]),
                None,
                XSD.string,
            ),
            "fail_position": (
                lambda: random.choice(["Open", "Closed", "Last"]),
                None,
                XSD.string,
            ),
            "disc_type": (
                lambda: random.choice(["Concentric", "Double-offset", "Triple-offset"]),
                None,
                XSD.string,
            ),
            "seat_material": (
                lambda: random.choice(["PTFE", "EPDM", "Metal", "NBR"]),
                None,
                XSD.string,
            ),
        },
        "TripleOffsetButterflyValve": {
            "torque_requirement": (lambda: random.uniform(100, 500), "Nm", XSD.decimal),
            "fire_safe_rating": (
                lambda: random.choice(["API 607", "API 6FA", "BS 6755", "None"]),
                None,
                XSD.string,
            ),
            "metal_seat_hardness": (
                lambda: random.uniform(400, 900),
                "HV",
                XSD.decimal,
            ),
        },
        "BallControlValve": {
            "cv_value": (lambda: random.uniform(15, 800), None, XSD.decimal),
            "rangeability": (lambda: random.uniform(50, 150), None, XSD.decimal),
            "leakage_class": (
                lambda: random.choice(["IV", "V", "VI"]),
                None,
                XSD.string,
            ),
            "fail_position": (
                lambda: random.choice(["Open", "Closed", "Last"]),
                None,
                XSD.string,
            ),
            "ball_type": (
                lambda: random.choice(
                    ["Full bore", "Reduced bore", "V-notch", "Segmented"]
                ),
                None,
                XSD.string,
            ),
            "port_size": (
                lambda: random.choice(["Full", "Regular", "Reduced"]),
                None,
                XSD.string,
            ),
        },
        "VNotchBallValve": {
            "v_notch_angle": (
                lambda: random.choice(["30", "60", "90"]),
                "degrees",
                XSD.string,
            ),
            "flow_characteristic": (
                lambda: random.choice(
                    ["Equal Percentage", "Modified Equal Percentage"]
                ),
                None,
                XSD.string,
            ),
        },
        "TrunnionMountedBallValve": {
            "trunnion_material": (
                lambda: random.choice(["SS316", "SS17-4PH", "Duplex SS", "Inconel"]),
                None,
                XSD.string,
            ),
            "seat_injection_system": (
                lambda: random.choice(["None", "Single", "Double"]),
                None,
                XSD.string,
            ),
            "mounting_style": (
                lambda: random.choice(["Side Entry", "Top Entry", "Welded"]),
                None,
                XSD.string,
            ),
        },
        # Sensor Types
        "TemperatureSensor": {
            "measured_value": (lambda: random.uniform(0, 100), "°C", XSD.decimal),
            "accuracy": (lambda: random.uniform(0.1, 0.5), "°C", XSD.decimal),
            "response_time": (lambda: random.uniform(1, 5), "s", XSD.decimal),
            "sensor_type": (
                lambda: random.choice(["RTD", "Thermocouple", "Thermistor"]),
                None,
                XSD.string,
            ),
        },
        "RTDSensor": {
            "measured_value": (lambda: random.uniform(-50, 200), "°C", XSD.decimal),
            "accuracy": (lambda: random.uniform(0.05, 0.3), "°C", XSD.decimal),
            "response_time": (lambda: random.uniform(0.8, 4), "s", XSD.decimal),
            "resistance_at_0c": (
                lambda: random.choice(["100", "1000"]),
                "Ohm",
                XSD.string,
            ),
            "wire_configuration": (
                lambda: random.choice(["2-wire", "3-wire", "4-wire"]),
                None,
                XSD.string,
            ),
            "element_material": (
                lambda: random.choice(["Platinum", "Nickel", "Copper"]),
                None,
                XSD.string,
            ),
        },
        "PT100Sensor": {
            "measured_value": (lambda: random.uniform(-50, 200), "°C", XSD.decimal),
            "accuracy": (lambda: random.uniform(0.03, 0.15), "°C", XSD.decimal),
            "response_time": (lambda: random.uniform(0.5, 3), "s", XSD.decimal),
            "resistance_at_0c": (lambda: "100", "Ohm", XSD.string),
            "wire_configuration": (
                lambda: random.choice(["2-wire", "3-wire", "4-wire"]),
                None,
                XSD.string,
            ),
            "iec_class": (
                lambda: random.choice(["Class A", "Class B", "Class AA", "Class 1/3B"]),
                None,
                XSD.string,
            ),
        },
        "ThermocoupleType": {
            "measured_value": (lambda: random.uniform(-100, 1300), "°C", XSD.decimal),
            "accuracy": (lambda: random.uniform(0.5, 2.0), "°C", XSD.decimal),
            "response_time": (lambda: random.uniform(0.1, 1.0), "s", XSD.decimal),
            "thermocouple_type": (
                lambda: random.choice(["J", "K", "T", "E", "S", "R", "B", "N"]),
                None,
                XSD.string,
            ),
            "junction_type": (
                lambda: random.choice(["Grounded", "Ungrounded", "Exposed"]),
                None,
                XSD.string,
            ),
        },
        "TypeKThermocouple": {
            "measured_value": (lambda: random.uniform(-200, 1300), "°C", XSD.decimal),
            "accuracy": (lambda: random.uniform(0.4, 1.5), "°C", XSD.decimal),
            "response_time": (lambda: random.uniform(0.1, 0.8), "s", XSD.decimal),
            "thermocouple_type": (lambda: "K", None, XSD.string),
            "junction_type": (
                lambda: random.choice(["Grounded", "Ungrounded", "Exposed"]),
                None,
                XSD.string,
            ),
            "positive_leg_material": (lambda: "Nickel-Chromium", None, XSD.string),
            "negative_leg_material": (lambda: "Nickel-Alumel", None, XSD.string),
        },
        "PressureSensor": {
            "measured_value": (lambda: random.uniform(0, 20), "bar", XSD.decimal),
            "accuracy": (lambda: random.uniform(0.1, 1.0), "%", XSD.decimal),
            "response_time": (lambda: random.uniform(0.1, 2), "s", XSD.decimal),
            "sensor_type": (
                lambda: random.choice(["Gauge", "Absolute", "Differential"]),
                None,
                XSD.string,
            ),
        },
        "StrainGaugeSensor": {
            "measured_value": (lambda: random.uniform(0, 15), "bar", XSD.decimal),
            "accuracy": (lambda: random.uniform(0.05, 0.5), "%", XSD.decimal),
            "response_time": (lambda: random.uniform(0.05, 1.0), "s", XSD.decimal),
            "sensor_type": (
                lambda: random.choice(["Gauge", "Absolute", "Differential"]),
                None,
                XSD.string,
            ),
            "bridge_configuration": (
                lambda: random.choice(["Quarter Bridge", "Half Bridge", "Full Bridge"]),
                None,
                XSD.string,
            ),
            "excitation_voltage": (lambda: random.uniform(2, 12), "V", XSD.decimal),
        },
        # Controller Types
        "PIDController": {
            "kp": (lambda: random.uniform(0.1, 10), None, XSD.decimal),
            "ki": (lambda: random.uniform(0.01, 1), None, XSD.decimal),
            "kd": (lambda: random.uniform(0, 0.1), None, XSD.decimal),
            "control_mode": (
                lambda: random.choice(["Auto", "Manual", "Cascade"]),
                None,
                XSD.string,
            ),
        },
        "SingleLoopController": {
            "kp": (lambda: random.uniform(0.1, 10), None, XSD.decimal),
            "ki": (lambda: random.uniform(0.01, 1), None, XSD.decimal),
            "kd": (lambda: random.uniform(0, 0.1), None, XSD.decimal),
            "control_mode": (
                lambda: random.choice(["Auto", "Manual"]),
                None,
                XSD.string,
            ),
            "input_type": (
                lambda: random.choice(["4-20mA", "0-10V", "Thermocouple", "RTD"]),
                None,
                XSD.string,
            ),
            "output_type": (
                lambda: random.choice(["4-20mA", "0-10V", "Relay", "SSR"]),
                None,
                XSD.string,
            ),
        },
        "CascadeController": {
            "kp_primary": (lambda: random.uniform(0.1, 5), None, XSD.decimal),
            "ki_primary": (lambda: random.uniform(0.01, 0.5), None, XSD.decimal),
            "kd_primary": (lambda: random.uniform(0, 0.05), None, XSD.decimal),
            "kp_secondary": (lambda: random.uniform(0.5, 8), None, XSD.decimal),
            "ki_secondary": (lambda: random.uniform(0.05, 0.8), None, XSD.decimal),
            "kd_secondary": (lambda: random.uniform(0, 0.08), None, XSD.decimal),
            "control_mode": (lambda: "Cascade", None, XSD.string),
        },
        # Heat Exchanger Types
        "CoolingTower": {
            "heat_rejection": (lambda: random.uniform(1000, 5000), "kW", XSD.decimal),
            "approach_temp": (lambda: random.uniform(3, 8), "°C", XSD.decimal),
            "air_flow": (lambda: random.uniform(50000, 200000), "m3/h", XSD.decimal),
            "tower_type": (
                lambda: random.choice(
                    ["Forced draft", "Induced draft", "Natural draft"]
                ),
                None,
                XSD.string,
            ),
        },
    }

    g.add((instance_uri, RDF.type, ISO15926[equipment_type]))
    print(f"Added type: {instance_uri} is a {equipment_type}")

    # Add tag number with proper datatype
    tag = f"{tag_prefix}-{str(random.randint(1000, 9999))}"
    g.add((instance_uri, ISO15926.hasTag, Literal(tag, datatype=XSD.string)))
    print(f"Added tag: {tag}")

    # Add property definitions first
    if equipment_type in properties:
        print(f"Adding properties for {equipment_type}")
        for prop_name, (value_func, unit, datatype) in properties[
            equipment_type
        ].items():
            # Create property instance
            prop_id = str(uuid.uuid4())
            prop_uri = DATA[prop_id]
            prop_type_uri = property_ns[prop_name]

            # Property triples
            value = value_func()
            print(f"Adding property {prop_name} with value {value}")

            # Add the property relationship
            g.add((instance_uri, has_property, prop_uri))
            g.add((prop_uri, RDF.type, prop_type_uri))
            g.add((prop_uri, has_value, Literal(value, datatype=datatype)))

            # Add unit if present
            if unit:
                g.add((prop_uri, has_unit, Literal(unit, datatype=XSD.string)))

            # Debug output
            print(f"Added triples for {prop_name}:")
            print(f"  {instance_uri} {has_property} {prop_uri}")
            print(f"  {prop_uri} rdf:type {prop_type_uri}")
            print(f"  {prop_uri} {has_value} {value}")
            if unit:
                print(f"  {prop_uri} {has_unit} {unit}")

    return instance_uri, tag


def generate_control_loop(g, sensor, controller, actuator):
    """Generate control loop relationships"""
    loop_id = str(uuid.uuid4())
    loop_uri = DATA[f"loop-{loop_id}"]

    g.add((loop_uri, RDF.type, ISO15926.ControlLoop))
    g.add((loop_uri, ISO15926.hasSensor, sensor))
    g.add((loop_uri, ISO15926.hasController, controller))
    g.add((loop_uri, ISO15926.hasActuator, actuator))

    return loop_uri


def generate_cooling_system(g):
    """Generate a cooling system with multiple equipment instances"""
    print("Generating cooling system RDF data...")
    system_id = str(uuid.uuid4())
    system_uri = DATA[system_id]

    # Define relationships
    contains = ISO15926["contains"]
    controls = ISO15926["controls"]
    is_connected_to = ISO15926["isConnectedTo"]
    monitors = ISO15926["monitors"]

    # Create system
    g.add((system_uri, RDF.type, ISO15926["CoolingSystem"]))

    # Create specific equipment instances with expanded class types

    # Different types of pumps
    print("Creating pumps...")
    instance_uris = []

    # Basic pump types
    primary_pump, primary_pump_tag = generate_equipment_instance(
        g, "SingleStagePump", "PPMP"
    )
    secondary_pump, secondary_pump_tag = generate_equipment_instance(
        g, "MultiStagePump", "SPMP"
    )

    # Advanced pump types from deeper hierarchy
    cooling_tower_pump, ct_pump_tag = generate_equipment_instance(
        g, "EndSuctionPump", "CTPMP"
    )
    chilled_water_pump, chw_pump_tag = generate_equipment_instance(
        g, "CirculatorPump", "CHWPMP"
    )
    condenser_pump, cond_pump_tag = generate_equipment_instance(
        g, "APIProcessPump", "CDPMP"
    )
    glycol_pump, glycol_pump_tag = generate_equipment_instance(
        g, "WetRotorCirculator", "GLYPMP"
    )
    chemical_pump, chem_pump_tag = generate_equipment_instance(
        g, "ChemicalProcessPump", "CHMPMP"
    )
    oil_pump, oil_pump_tag = generate_equipment_instance(
        g, "ExternalGearPump", "OILPMP"
    )

    # Different types of valves
    print("Creating valves...")
    # Basic valve
    main_valve, main_valve_tag = generate_equipment_instance(g, "ControlValve", "MCV")

    # Deeper valve hierarchy
    temp_control_valve, tcv_tag = generate_equipment_instance(
        g, "GlobeControlValve", "TCV"
    )
    bypass_valve, bypass_tag = generate_equipment_instance(
        g, "AngleBodyGlobeValve", "BPV"
    )
    flow_valve, flow_tag = generate_equipment_instance(g, "VNotchBallValve", "FCV")
    pressure_valve, prs_tag = generate_equipment_instance(
        g, "TrunnionMountedBallValve", "PCV"
    )
    high_noise_valve, hnv_tag = generate_equipment_instance(
        g, "NoiseReductionTrimValve", "HNV"
    )

    # Different types of sensors
    print("Creating sensors...")
    # Temperature sensors with deeper hierarchy
    temp_sensor1, temp1_tag = generate_equipment_instance(g, "PT100Sensor", "TST")
    temp_sensor2, temp2_tag = generate_equipment_instance(g, "TypeKThermocouple", "TTK")

    # Pressure sensors with deeper hierarchy
    pressure_sensor1, prs1_tag = generate_equipment_instance(
        g, "StrainGaugeSensor", "PST"
    )
    pressure_sensor2, prs2_tag = generate_equipment_instance(g, "PressureSensor", "PSG")

    # Flow sensor
    flow_sensor, flow_sensor_tag = generate_equipment_instance(g, "FlowSensor", "FST")

    # Different types of controllers
    print("Creating controllers...")
    # Controllers with deeper hierarchy
    main_controller, main_ctrl_tag = generate_equipment_instance(
        g, "SingleLoopController", "TIC"
    )
    cascade_controller, cascade_ctrl_tag = generate_equipment_instance(
        g, "CascadeController", "CIC"
    )

    # Create a heat exchanger
    print("Creating heat exchangers...")
    cooling_tower, ct_tag = generate_equipment_instance(g, "CoolingTower", "CT")
    heat_exchanger, hx_tag = generate_equipment_instance(g, "HeatExchanger", "HX")

    # Add all equipment to the system
    equipment_list = [
        primary_pump,
        secondary_pump,
        cooling_tower_pump,
        chilled_water_pump,
        condenser_pump,
        glycol_pump,
        chemical_pump,
        oil_pump,
        main_valve,
        temp_control_valve,
        bypass_valve,
        flow_valve,
        pressure_valve,
        high_noise_valve,
        temp_sensor1,
        temp_sensor2,
        pressure_sensor1,
        pressure_sensor2,
        flow_sensor,
        main_controller,
        cascade_controller,
        cooling_tower,
        heat_exchanger,
    ]

    for equipment in equipment_list:
        g.add((system_uri, contains, equipment))
        print(f"Added {equipment} to system {system_uri}")

    # Create complex control relationships
    print("Creating control relationships...")

    # Temperature control loop
    g.add((temp_sensor1, monitors, heat_exchanger))
    g.add((main_controller, controls, temp_control_valve))
    g.add((temp_sensor1, is_connected_to, main_controller))

    # Cascade control loop
    g.add((temp_sensor2, monitors, cooling_tower))
    g.add((pressure_sensor1, monitors, primary_pump))
    g.add((cascade_controller, controls, primary_pump))
    g.add((temp_sensor2, is_connected_to, cascade_controller))
    g.add((pressure_sensor1, is_connected_to, cascade_controller))

    # Flow control loop
    g.add((flow_sensor, monitors, chilled_water_pump))
    g.add((main_controller, controls, flow_valve))
    g.add((flow_sensor, is_connected_to, main_controller))

    # Equipment connections
    g.add((primary_pump, is_connected_to, heat_exchanger))
    g.add((heat_exchanger, is_connected_to, secondary_pump))
    g.add((secondary_pump, is_connected_to, temp_control_valve))
    g.add((temp_control_valve, is_connected_to, cooling_tower))
    g.add((cooling_tower, is_connected_to, cooling_tower_pump))
    g.add((cooling_tower_pump, is_connected_to, primary_pump))

    # Advanced piping system with more equipment
    g.add((chilled_water_pump, is_connected_to, bypass_valve))
    g.add((bypass_valve, is_connected_to, condenser_pump))
    g.add((condenser_pump, is_connected_to, pressure_valve))
    g.add((pressure_valve, is_connected_to, high_noise_valve))
    g.add((high_noise_valve, is_connected_to, chemical_pump))
    g.add((chemical_pump, is_connected_to, oil_pump))
    g.add((oil_pump, is_connected_to, glycol_pump))
    g.add((glycol_pump, is_connected_to, cooling_tower))

    # Add industry standard compliance relations where applicable
    standard_uri = ISO15926["API610Standard"]
    g.add((standard_uri, RDF.type, ISO15926["IndustryStandard"]))
    g.add((condenser_pump, ISO15926["compliesWith"], standard_uri))

    # Add operation modes for advanced controllers
    normal_mode_uri = ISO15926["NormalMode"]
    emergency_mode_uri = ISO15926["EmergencyMode"]
    g.add((normal_mode_uri, RDF.type, ISO15926["OperationMode"]))
    g.add((emergency_mode_uri, RDF.type, ISO15926["OperationMode"]))
    g.add((system_uri, ISO15926["hasOperationMode"], normal_mode_uri))
    g.add((system_uri, ISO15926["hasOperationMode"], emergency_mode_uri))

    # Add maintenance information for critical equipment
    g.add(
        (
            primary_pump,
            ISO15926["maintenanceInterval"],
            Literal("3 months", datatype=XSD.string),
        )
    )
    g.add(
        (
            cooling_tower,
            ISO15926["maintenanceInterval"],
            Literal("6 months", datatype=XSD.string),
        )
    )

    print("Cooling system RDF data generated successfully.")
    return system_uri


def add_industry_standards(g):
    """Add industry standards and specialized relationships to the ontology"""
    print("Adding industry standards and specialized relationships...")

    # Create a namespace for standards
    std_ns = Namespace("http://standards.iso.org/iso/15926/-4/standards/")

    # Define industry standards
    standards = {
        "API610Standard": "API 610 - Centrifugal Pumps for Petroleum, Petrochemical and Natural Gas Industries",
        "API676Standard": "API 676 - Positive Displacement Pumps - Rotary",
        "ASME_B73_1": "ASME B73.1 - Specification for Horizontal End Suction Centrifugal Pumps",
        "ASME_B73_2": "ASME B73.2 - Specification for Vertical In-Line Centrifugal Pumps",
        "IEC60534": "IEC 60534 - Industrial-Process Control Valves",
        "ISA75_01": "ISA-75.01 - Flow Equations for Sizing Control Valves",
        "IEC61508": "IEC 61508 - Functional Safety of Electrical/Electronic/Programmable Electronic Safety-Related Systems",
        "ISO13709": "ISO 13709 - Centrifugal pumps for petroleum, petrochemical and natural gas industries",
        "ANSI_FCI_70_2": "ANSI/FCI 70-2 - Control Valve Seat Leakage",
        "IEC60751": "IEC 60751 - Industrial Platinum Resistance Thermometers",
        "ASME_PTC_23": "ASME PTC 23 - Atmospheric Water Cooling Equipment",
    }

    # Create standard instances
    for std_id, std_desc in standards.items():
        std_uri = ISO15926[std_id]
        g.add((std_uri, RDF.type, ISO15926["IndustryStandard"]))
        g.add((std_uri, RDFS.label, Literal(std_desc, datatype=XSD.string)))
        print(f"Added industry standard: {std_id} - {std_desc}")

    # Define specialized relationships
    relationships = [
        "compliesWith",
        "certifiedBy",
        "hasMaintenanceSchedule",
        "hasFailureMode",
        "hasSparepart",
        "hasOperatingRange",
        "hasWarranty",
        "manufacturedBy",
        "installedBy",
        "lastInspectedOn",
        "requiresCalibration",
        "hasTestCertificate",
        "controlsTemperature",
        "controlsPressure",
        "controlsFlow",
        "hasNominalCapacity",
        "hasEfficiency",
        "hasMaterialCompatibility",
    ]

    # Create relationship properties
    for rel in relationships:
        rel_uri = ISO15926[rel]
        g.add((rel_uri, RDF.type, RDF.Property))
        print(f"Added relationship: {rel}")

    # Create operation modes
    operation_modes = [
        "NormalMode",
        "ShutdownMode",
        "StartupMode",
        "EmergencyMode",
        "MaintenanceMode",
        "LowLoadMode",
        "HighLoadMode",
    ]

    for mode in operation_modes:
        mode_uri = ISO15926[mode]
        g.add((mode_uri, RDF.type, ISO15926["OperationMode"]))
        print(f"Added operation mode: {mode}")

    # Add hasOperationMode relationship
    g.add((ISO15926["hasOperationMode"], RDF.type, RDF.Property))
    print("Added hasOperationMode relationship")

    print("Industry standards and specialized relationships added.")


def add_domain_knowledge_rules(g):
    """Add domain knowledge rules to enhance the dataset for RAG systems"""
    print("Adding domain knowledge rules...")

    # Define domain knowledge using RDFS.comment to store textual knowledge
    domain_knowledge = {
        "CentrifugalPump": "Centrifugal pumps use an impeller to increase fluid pressure and flow. They are commonly used for water circulation and low-viscosity fluids.",
        "MultiStagePump": "Multi-stage pumps contain multiple impellers in series to achieve higher discharge pressures. Ideal for high-pressure, high-head applications.",
        "SingleStagePump": "Single-stage pumps have one impeller and are used for low to medium pressure applications with high flow rates.",
        "EndSuctionPump": "End suction pumps draw fluid from a single inlet aligned with the pump shaft. They are widely used in water supply, irrigation, and HVAC systems.",
        "CirculatorPump": "Circulators are small, low-power pumps primarily used for HVAC and hot water recirculation systems in residential and light commercial applications.",
        "PositiveDisplacementPump": "Positive displacement pumps trap fluid in a cavity and force it to the discharge port. They excel at handling high-viscosity fluids and maintaining precise flow rates.",
        "GearPump": "Gear pumps use rotating gears to move fluid. They provide smooth, pulse-free flow and are suitable for high-pressure applications with medium to high-viscosity fluids.",
        "ControlValve": "Control valves regulate flow or pressure by changing the valve opening in response to signals from controllers. They are critical components in process control systems.",
        "GlobeControlValve": "Globe valves provide precise flow control and good shutoff capabilities. They are commonly used for throttling service in process control applications.",
        "BallControlValve": "Ball control valves use a perforated ball to control flow. They offer tight shutoff, low torque requirements, and are excellent for high-pressure applications.",
        "RTDSensor": "RTD (Resistance Temperature Detector) sensors measure temperature by correlating the resistance of the RTD element with temperature. They provide high accuracy and stability.",
        "PT100Sensor": "PT100 sensors are a specific type of RTD with a resistance of 100 ohms at 0°C. They offer excellent accuracy and are widely used in industrial applications.",
        "ThermocoupleType": "Thermocouples measure temperature using the thermoelectric effect, generating a voltage proportional to temperature difference. They are simple, rugged, and can measure a wide temperature range.",
        "PIDController": "PID controllers use Proportional-Integral-Derivative control algorithms to maintain a setpoint by continuously calculating an error value and applying corrections.",
        "CascadeController": "Cascade control uses multiple control loops where the output of the primary controller sets the setpoint for the secondary controller, improving response to disturbances.",
        "CoolingTower": "Cooling towers reject heat from water-cooled systems to the atmosphere through evaporation. They are critical in power plants, HVAC systems, and industrial processes.",
    }

    # Add the domain knowledge as comments
    for class_name, knowledge in domain_knowledge.items():
        class_uri = ISO15926[class_name]
        g.add((class_uri, RDFS.comment, Literal(knowledge, datatype=XSD.string)))
        print(f"Added domain knowledge for: {class_name}")

    # Add compatibility rules
    compatibility_rules = [
        (
            ISO15926["PositiveDisplacementPump"],
            ISO15926["requiresPropertyValue"],
            Literal(
                "When handling high-viscosity fluids above 500 cSt, positive displacement pumps are recommended over centrifugal pumps.",
                datatype=XSD.string,
            ),
        ),
        (
            ISO15926["CentrifugalPump"],
            ISO15926["requiresPropertyValue"],
            Literal(
                "NPSH available must exceed NPSH required by at least 0.5m to prevent cavitation.",
                datatype=XSD.string,
            ),
        ),
        (
            ISO15926["GlobeControlValve"],
            ISO15926["hasMaterialCompatibility"],
            Literal(
                "For corrosive fluids with pH < 4, use valves with Monel, Hastelloy, or lined bodies.",
                datatype=XSD.string,
            ),
        ),
        (
            ISO15926["BallControlValve"],
            ISO15926["hasOperatingRange"],
            Literal(
                "Maximum operating temperature for PTFE seats is 180°C.",
                datatype=XSD.string,
            ),
        ),
        (
            ISO15926["TypeKThermocouple"],
            ISO15926["hasOperatingRange"],
            Literal(
                "Typical measurement range is -200°C to 1350°C with standard limits of error of ±2.2°C or ±0.75%.",
                datatype=XSD.string,
            ),
        ),
    ]

    # Add the compatibility rules
    for subject, predicate, obj in compatibility_rules:
        g.add((subject, predicate, obj))
        print(f"Added compatibility rule between {subject} and {obj}")

    # Add links to the property requirements
    g.add((ISO15926["requiresPropertyValue"], RDF.type, RDF.Property))
    g.add((ISO15926["hasMaterialCompatibility"], RDF.type, RDF.Property))
    g.add((ISO15926["hasOperatingRange"], RDF.type, RDF.Property))

    print("Domain knowledge rules added.")


def setup_and_generate():
    """Setup RDF graph, add ontology, and generate cooling system data"""
    # Create RDF graph
    g = Graph()

    # Bind prefixes
    g.bind("iso15926", ISO15926)
    g.bind("data", DATA)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # Add class hierarchy
    add_class_hierarchy(g)

    # Add property definitions
    add_property_definitions(g)

    # Add industry standards and specialized relationships
    add_industry_standards(g)

    # Add domain knowledge rules
    add_domain_knowledge_rules(g)

    # Generate cooling system and equipment
    generate_cooling_system(g)

    # Serialize to different formats
    g.serialize(destination="cooling_system.ttl", format="turtle")
    g.serialize(destination="cooling_system.nt", format="nt")

    # Generate SQL load script for Virtuoso
    with open("cooling_system_load.sql", "w") as f:
        f.write("-- SQL script to load RDF data into Virtuoso\n")
        f.write("SPARQL CLEAR GRAPH <http://example.org/cooling-system>;\n")
        f.write("DELETE FROM DB.DBA.LOAD_LIST WHERE ll_file = 'cooling_system.ttl';\n")
        f.write("LD_ADD ('cooling_system.ttl', 'http://example.org/cooling-system');\n")
        f.write("RDFLOADER_RUN();\n")
        f.write("CHECKPOINT;\n")

    print(
        "RDF data has been generated and saved to cooling_system.ttl and cooling_system.nt"
    )
    print("SQL load script has been generated as cooling_system_load.sql")


if __name__ == "__main__":
    setup_and_generate()
