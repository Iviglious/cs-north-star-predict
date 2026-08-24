from pydantic import BaseModel, Field, model_validator

class ChurnInput(BaseModel):
    # Numeric fields with constraints
    sla_target_hours: int = Field(..., ge=0)

    # Categorical fields, with multiple string values
    channel: str = Field(default="email", pattern="^(email|phone|in_app|webchat)$")
    priority: str = Field(default="Medium", pattern="^(Low|Medium|High|Urgent)$")
    plan_tier: str = Field(default="Free", pattern="^(Free|Standard|Pro|Enterprise)$")
    sentiment: str = Field(default="Neutral", pattern="^(Positive|Neutral|Negative)$")

    # Text fields
    tags: str = Field(default="vat;subscription;invoice", max_length=100)

"""
    # This validator runs before the model is fully constructed
    # It receives the raw input values as a dictionary
    # model_validator allows multiple fields to be checked
    # If we only wanted to check one field we could use field_validator
    # @classmethod passes the class to our validator
    @model_validator(mode='before')
    @classmethod
    def check_charge_ratios(cls, values):
        # Check that both required fields are present and non-zero
        if values.get("total_day_minutes") and values.get("total_day_charge"):
            try:
                # Compute the ratio of charge to minutes
                ratio = values["total_day_charge"] / values["total_day_minutes"]
                # If the cost per minute is unusually high, raise an error
                if ratio > 0.5:
                    raise ValueError("Day charge seems too high for the minutes used.")
            except ZeroDivisionError:
                # Fail silently if total_day_minutes is zero to avoid division error
                pass
        # Return the validated (or original) dictionary
        return values
"""