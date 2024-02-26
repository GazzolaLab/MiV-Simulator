from pydantic import BaseModel

PopulationName = str
PostSynapticPopulationName = PopulationName
PreSynapticPopulationName = PopulationName


class CellsMetaData(BaseModel):
    """Cells metadata"""

    population_names: list[PopulationName]
    population_ranges: dict[PopulationName, tuple[int, int]]
    cell_attribute_info: dict[PopulationName, dict[str, list[str]]]

    def has(self, population: PopulationName, attribute: str) -> bool:
        return attribute in self.cell_attribute_info.get(population, {})


class Tree(BaseModel):
    """TBD"""


class Projection(BaseModel):
    """TBD"""
