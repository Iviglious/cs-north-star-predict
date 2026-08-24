from pydantic import BaseModel, Field, model_validator

class ChurnInput(BaseModel):
    # Mandatory field (due to ...), int greater than or equal to 0
    account_length: int = Field(..., ge=0)
    
    # Must be and int in the range 0, 1
    international_plan: int = Field(..., ge=0, le=1)
    voice_mail_plan: int = Field(..., ge=0, le=1)
    number_vmail_messages: int = Field(..., ge=0)
    total_day_minutes: float = Field(..., ge=0)
    total_day_calls: int = Field(..., ge=0)
    total_day_charge: float = Field(..., ge=0)
    total_eve_minutes: float = Field(..., ge=0)
    total_eve_calls: int = Field(..., ge=0)
    total_eve_charge: float = Field(..., ge=0)
    total_night_minutes: float = Field(..., ge=0)
    total_night_calls: int = Field(..., ge=0)
    total_night_charge: float = Field(..., ge=0)
    total_intl_minutes: float = Field(..., ge=0)
    total_intl_calls: int = Field(..., ge=0)
    total_intl_charge: float = Field(..., ge=0)
    number_customer_service_calls: int = Field(..., ge=0)
    
    
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