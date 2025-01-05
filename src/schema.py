from typing import Literal, Union, List, Optional

from pydantic import BaseModel, Field

class BaseTrait(BaseModel):
    trait_index: int
    trait: str = Field(description="The sematic description of individual traits")

class BaseCriterion(BaseModel):
    index: int
    in_or_ex: Literal["IN", "EX"] = Field(description="Indicates inclusion ('IN') or exclusion ('EX').")
    original_criterion: str = Field(description="The original eligibility criterion.")

class TraitDecomposed(BaseTrait):
    computable: bool = Field(description="Whether the trait can be found in the OMOP CDM.")
    negation: bool = Field(description="Whether the trait contains negation.")
    rephrased_trait: Union[str|None] = Field(description="Positive rephrasing of the trait if it contains negation.")
    modifier: Union[str|None] = Field(description="Modifiers that make the main entity unclear.")
    main_entity_content: str = Field(description="The main entity of the trait.")
    main_entity_type: Literal["Condition", "Measurement", "Procedure", "Drug", "Observation", "Visit", "Demographic", "Device", "Other"] = Field(description="Type of the main entity.")
    constraint_detail: Union[List[str]|None] = Field(description="constrain of the main_entity_content")
    constraint_type: Union[List[Literal["Value", "Time", "Duration time", "Count", "Form", "Dosage", "Route", "Age", "Race", "Gender", "Ethnicity", "Other"]] | None] = Field(description="Type of the constraint.")

class CriterionWTraits(BaseCriterion):
    computable: Union[Literal["Completely", "Partially", "Not at all"], bool] = Field(description="Whether the Patient meet the criterion can be found in the OMOP CDM.")
    traits: List[BaseTrait] = Field(description="Individual traits derived from the criterion.")
    logic_relation: Union[str|None] = Field(description="Logical relationship between traits if more than one (e.g., '1 AND 2'). Null if only one trait.")

class CriterionWDecomposedTrait(CriterionWTraits):
    traits: List[TraitDecomposed] = Field(description="Individual traits derived from the criterion.")