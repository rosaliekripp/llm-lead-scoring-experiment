# Data

This directory contains the input data used in the LLM lead scoring experiment.

## Dataset

`synthetic/synthetic_lead_profiles.csv` contains five synthetic B2B lead profiles. These profiles define the baseline conditions from which the experimental perturbations are generated.

Each profile represents a prospective organization in a fictional sales scenario. The dataset contains no real leads, customer records, or personal data.

## Data Structure

The study uses synthetic data only. No real CRM data are included.

The five baseline profiles represent different combinations of firmographic fit and behavioral engagement:

- **Profile A — Classic Prospect:** Moderate values across all features.
- **Profile B — Cold Enterprise:** Strong firmographic fit and low engagement.
- **Profile C — Hot Small Lead:** Weak firmographic fit and high engagement.
- **Profile D — Growing Mid-Market:** Moderate values distinct from Profile A.
- **Profile E — Long Shot:** Low values across both feature groups.

During the experiment, each baseline profile is extended with twelve one-factor-at-a-time perturbations: every feature is independently changed to a predefined low or high value.

If the experiment is extended with real-world lead data, these data may be stored in a separate `real/` subdirectory alongside `synthetic/`. Any such data must be collected, anonymized, processed, and shared in accordance with applicable privacy, legal, ethical, and organizational requirements.

## Variable Description

| Variable | Type | Feature group | Description |
|---|---|---|---|
| `profile_id` | Categorical | Identifier | Unique identifier of the reference profile. |
| `company_size` | Numeric | Firmographic | Number of employees in the organization. |
| `industry` | Categorical | Firmographic | Industry in which the organization operates. |
| `region` | Categorical | Firmographic | Geographic market of the organization. |
| `website_dwell_time` | Numeric | Behavioral | Cumulative time spent on the vendor's website. |
| `demo_requests` | Numeric | Behavioral | Number of requested product demonstrations. |
| `email_response` | Free text | Behavioral | Response to an outbound contact attempt. |

The behavioral variables describe interactions attributed to a prospective organization within the synthetic scenario. They do not describe the characteristics or behavior of a real person.