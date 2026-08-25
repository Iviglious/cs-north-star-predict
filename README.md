# cs-north-star-predict


## Start gradio
### Install
```
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### Start

```
# Assign Team Predictor
python gradio_main.py
https://2e2b134ec08e5a4865.gradio.live 

# Triage Assistant
python lessons/app.py --host 0.0.0.0 --port 7861 --share=True
https://2da0952919df00da89.gradio.live/
```

## Data Fields
```
case_id
snapshot_at
created_at
channel
case_type
category
subcategory
priority
sla_target_hours
first_response_time_hours
resolution_time_hours
status
resolution_code
escalated
assigned_team
escalation_team
customer_tenure_months
plan_tier
region_uk
age_band
gender
case_summary
sentiment
csat_score
tags
```


## Cases to check

Case ID: ND-2025-002119
Predicted team: engineering
Actual team: data
Probabilities:
  billing: 0.0103
  data: 0.3672
  engineering: 0.6100
  operations: 0.0010
  security: 0.0082
  support: 0.0031

Case ID: ND-2025-002078
Predicted team: operations
Actual team: support
Probabilities:
  billing: 0.0164
  data: 0.0294
  engineering: 0.0070
  operations: 0.5700
  security: 0.0149
  support: 0.3623

Case ID: ND-2025-002050
Predicted team: security
Actual team: support
Probabilities:
  billing: 0.0002
  data: 0.0000
  engineering: 0.0002
  operations: 0.0037
  security: 0.9955
  support: 0.0003
