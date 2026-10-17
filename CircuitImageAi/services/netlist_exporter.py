import datetime
import uuid
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Optional

from core.utils import eng_suffix


class BaseNetlistExporter(ABC):
    @abstractmethod
    def generate(self, design_data: dict, lib: str, cell: str, view: str) -> str:
        pass


def _components_for_design(design_data: dict) -> list[dict]:
    components = [{
        "ref": "V1",
        "value": "DC 0 AC 1",
        "lib": "Simulation_SPICE",
        "part": "VDC",
        "pins": [("in", "1"), ("0", "2")],
        "spice": "V1 in 0 DC 0 AC 1",
        "spectre": "V1 (in 0) vsource dc=0 acmag=1",
    }]
    design_type = design_data.get("type")

    def add(ref: str, value: str, part: str, nodes: list[str], spice: str, spectre: str,
            lib_name: str = "Device", pin_numbers: tuple[str, ...] = ("1", "2")) -> None:
        components.append({
            "ref": ref,
            "value": value,
            "lib": lib_name,
            "part": part,
            "pins": list(zip(nodes, pin_numbers)),
            "spice": spice,
            "spectre": spectre,
        })

    if design_type in {"rc_lp", "rc_hp"}:
        resistance = eng_suffix(design_data["R"], "R")
        capacitance = eng_suffix(design_data["C"], "C")
        if design_type == "rc_lp":
            add("R1", resistance, "R", ["in", "out"],
                f"R1 in out {resistance}", f"R1 (in out) resistor r={resistance}")
            add("C1", capacitance, "C", ["out", "0"],
                f"C1 out 0 {capacitance}", f"C1 (out 0) capacitor c={capacitance}")
        else:
            add("C1", capacitance, "C", ["in", "out"],
                f"C1 in out {capacitance}", f"C1 (in out) capacitor c={capacitance}")
            add("R1", resistance, "R", ["out", "0"],
                f"R1 out 0 {resistance}", f"R1 (out 0) resistor r={resistance}")
    elif design_type == "resonator":
        for index, stage in enumerate(design_data.get("stages", []), start=1):
            suffix = f"_{index}"
            r1 = eng_suffix(stage["R1"], "R")
            r2 = eng_suffix(stage["R2"], "R")
            c1 = eng_suffix(stage["C1"], "C")
            c2 = eng_suffix(stage["C2"], "C")
            add(f"E{index}", "1", "VCVS", [f"out{suffix}", "0", f"n2{suffix}", "0"],
                f"E{index} out{suffix} 0 n2{suffix} 0 1.0",
                f"E{index} (out{suffix} 0 n2{suffix} 0) vcvs gain=1.0",
                lib_name="Simulation_SPICE", pin_numbers=("1", "2", "3", "4"))
            add(f"R1{suffix}", r1, "R", ["in", f"n1{suffix}"],
                f"R1{suffix} in n1{suffix} {r1}",
                f"R1{suffix} (in n1{suffix}) resistor r={r1}")
            add(f"R2{suffix}", r2, "R", [f"n1{suffix}", f"n2{suffix}"],
                f"R2{suffix} n1{suffix} n2{suffix} {r2}",
                f"R2{suffix} (n1{suffix} n2{suffix}) resistor r={r2}")
            add(f"C1{suffix}", c1, "C", [f"n1{suffix}", f"out{suffix}"],
                f"C1{suffix} n1{suffix} out{suffix} {c1}",
                f"C1{suffix} (n1{suffix} out{suffix}) capacitor c={c1}")
            add(f"C2{suffix}", c2, "C", [f"n2{suffix}", "0"],
                f"C2{suffix} n2{suffix} 0 {c2}",
                f"C2{suffix} (n2{suffix} 0) capacitor c={c2}")

    return components


class SpectreNetlistExporter(BaseNetlistExporter):
    def generate(self, design_data: dict, lib: str, cell: str, view: str) -> str:
        design_type = design_data.get("type", "circuit")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"// Library name: {lib}",
            f"// Cell name: {cell}",
            f"// View name: {view}",
            f"// {design_type} {now}",
            "simulator lang=spectre",
            "global 0",
        ]
        lines.extend(component["spectre"] for component in _components_for_design(design_data))
        if design_type in {"rc_lp", "rc_hp"}:
            lines.append("save v(out)")
        elif design_type == "resonator":
            lines.extend(
                f"save v(out_{index})"
                for index, _ in enumerate(design_data.get("stages", []), start=1)
            )
        lines.append("ac1 ac start=1 stop=1M dec=50")
        return "\n".join(lines)


class SpiceNetlistExporter(BaseNetlistExporter):
    def generate(self, design_data: dict, lib: str, cell: str, view: str) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"* {design_data.get('type', 'circuit')} {now}",
            f"* Lib:{lib} Cell:{cell} View:{view}",
        ]
        lines.extend(component["spice"] for component in _components_for_design(design_data))
        lines.extend([".ac dec 50 1 1meg", ".end"])
        return "\n".join(lines)


class LTspiceNetlistExporter(SpiceNetlistExporter):
    def generate(self, design_data: dict, lib: str, cell: str, view: str) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"* {design_data.get('type', 'circuit')} {now} (LTspice)",
            f"* Lib:{lib} Cell:{cell} View:{view}",
        ]
        lines.extend(component["spice"] for component in _components_for_design(design_data))
        lines.extend([".ac dec 50 1 1meg", ".backanno", ".end"])
        return "\n".join(lines)


class KiCadNetlistExporter(BaseNetlistExporter):
    @staticmethod
    def _make_element(parent: ET.Element, tag: str, text: Optional[str] = None, **attributes: str) -> ET.Element:
        element = ET.SubElement(parent, tag, attributes)
        if text is not None:
            element.text = text
        return element

    def generate(self, design_data: dict, lib: str, cell: str, view: str) -> str:
        components = _components_for_design(design_data)
        root = ET.Element("export", {"version": "D"})
        design = ET.SubElement(root, "design")
        self._make_element(design, "source", f"{lib}:{cell}:{view}")
        self._make_element(design, "date", datetime.datetime.now().isoformat(timespec="seconds"))
        self._make_element(design, "tool", "CircuitImageAI")

        component_ids = {component["ref"]: uuid.uuid4().hex[:8].upper() for component in components}
        component_section = ET.SubElement(root, "components")
        for component in components:
            comp = ET.SubElement(component_section, "comp", {"ref": component["ref"]})
            self._make_element(comp, "value", component["value"])
            self._make_element(comp, "footprint", "")
            self._make_element(comp, "datasheet", "")
            ET.SubElement(comp, "libsource", {
                "lib": component["lib"],
                "part": component["part"],
            })
            ET.SubElement(comp, "sheetpath", {"names": "/", "tstamps": "/"})
            self._make_element(comp, "tstamps", component_ids[component["ref"]])

        libparts: dict[tuple[str, str], tuple[str, ...]] = {}
        for component in components:
            key = (component["lib"], component["part"])
            libparts[key] = tuple(pin for _, pin in component["pins"])
        libpart_section = ET.SubElement(root, "libparts")
        for (lib_name, part), pins in libparts.items():
            libpart = ET.SubElement(libpart_section, "libpart", {"lib": lib_name, "part": part})
            self._make_element(libpart, "description", part)
            ET.SubElement(libpart, "footprints")
            fields = ET.SubElement(libpart, "fields")
            self._make_element(fields, "field", part, name="Reference")
            self._make_element(fields, "field", part, name="Value")
            pin_section = ET.SubElement(libpart, "pins")
            for pin in pins:
                ET.SubElement(pin_section, "pin", {
                    "num": pin,
                    "name": "~",
                    "type": "passive",
                })

        libraries = ET.SubElement(root, "libraries")
        for lib_name in dict.fromkeys(component["lib"] for component in components):
            library = ET.SubElement(libraries, "library")
            self._make_element(library, "logical", lib_name)
            self._make_element(library, "uri", f"${{KICAD_SYMBOL_DIR}}/{lib_name}.kicad_sym")

        nets: dict[str, list[tuple[str, str]]] = {}
        for component in components:
            for net_name, pin in component["pins"]:
                nets.setdefault(net_name, []).append((component["ref"], pin))

        net_section = ET.SubElement(root, "nets")
        for code, (net_name, nodes) in enumerate(nets.items(), start=1):
            net = ET.SubElement(net_section, "net", {"code": str(code), "name": net_name})
            for ref, pin in nodes:
                ET.SubElement(net, "node", {"ref": ref, "pin": pin})

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode", xml_declaration=True)


class NetlistFactory:
    _exporters = {
        "spectre": SpectreNetlistExporter,
        "spice": SpiceNetlistExporter,
        "ltspice": LTspiceNetlistExporter,
        "kicad": KiCadNetlistExporter,
    }

    @classmethod
    def get_exporter(cls, fmt: str) -> BaseNetlistExporter:
        exporter_class = cls._exporters.get(fmt.strip().lower())
        if not exporter_class:
            raise ValueError(f"Unsupported netlist format: {fmt}")
        return exporter_class()
