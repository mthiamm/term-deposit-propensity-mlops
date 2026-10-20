"""Input contract for one customer. Shared by the API and the batch job."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["yes", "no"]


class Customer(BaseModel):
    """One customer's raw attributes, exactly as the bank would hold them."""

    model_config = ConfigDict(extra="forbid")  # reject unknown/typo'd fields

    age: int = Field(ge=18, le=120)
    job: Literal[
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ]
    marital: Literal["married", "single", "divorced"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: YesNo
    balance: int
    housing: YesNo
    loan: YesNo
    contact: Literal["cellular", "telephone", "unknown"]
    day: int = Field(ge=1, le=31)
    month: Literal[
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=-1)
    previous: int = Field(ge=0)
    poutcome: Literal["failure", "other", "success", "unknown"]
