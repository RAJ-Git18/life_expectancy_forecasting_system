from pydantic import BaseModel, Field


class PredictLifespanRequest(BaseModel):
    adult_mortality: float = Field(
        ...,
        description="Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)",
    )
    infant_deaths: int = Field(
        ..., description="Number of Infant Deaths per 1000 population"
    )
    alcohol: float = Field(
        ...,
        description="Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)",
    )
    percentage_expenditure: float = Field(
        ...,
        description="Expenditure on health as a percentage of Gross Domestic Product per capita(%)",
    )
    hepatitis_b: float = Field(
        ...,
        description="Hepatitis B (HepB) immunization coverage among 1-year-olds (%)",
    )
    measles: int = Field(
        ..., description="Measles - number of reported cases per 1000 population"
    )
    bmi: float = Field(..., description="Average Body Mass Index of entire population")
    under_five_deaths: int = Field(
        ..., description="Number of under-five deaths per 1000 population"
    )
    polio: float = Field(
        ..., description="Polio (Pol3) immunization coverage among 1-year-olds (%)"
    )
    total_expenditure: float = Field(
        ...,
        description="General government expenditure on health as a percentage of total government expenditure (%)",
    )
    diphtheria: float = Field(
        ...,
        description="Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)",
    )
    hiv_aids: float = Field(
        ..., description="Deaths per 1 000 live births HIV/AIDS (0-4 years)"
    )
    gdp: float = Field(..., description="Gross Domestic Product per capita (in USD)")
    population: float = Field(..., description="Population of the country")
    thinness_1_19_years: float = Field(
        ...,
        description="Prevalence of thinness among children and adolescents for Age 10 to 19 (%)",
    )
    thinness_5_9_years: float = Field(
        ..., description="Prevalence of thinness among children for Age 5 to 9(%)"
    )
    income_composition_of_resources: float = Field(
        ...,
        description="Human Development Index in terms of income composition of resources (index ranging from 0 to 1)",
    )
    schooling: float = Field(..., description="Number of years of Schooling(years)")
