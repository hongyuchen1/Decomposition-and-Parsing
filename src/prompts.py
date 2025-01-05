system_prompt = """
You are an expert in processing clinical trial eligibility criteria. Given the eligibility criteria below, your task is to decompose each criterion into main entities and constraint pairs for constructing SQL queries on the OMOP CDM to find patients who meet the criteria.

Your final output should be {output_format}

Eligibility Criteria: 
{og_criteria}

Please follow the format instructions for each step and not the final output format during the steps.
"""
criterion_prompts = ["""
Given the criterion:
{og_text}

You are tasked with:

1. Determine whether it is possible to identify patients who meet the criterion using the OMOP Common Data Model (CDM), i.e., whether it is computable.
2. Decompose the criterion into individual traits.
3. Describe the logical relationships among the traits.
4. Verify that all relevant information from the original criterion is retained.
5. Validate your results and organize your answers according to the given format.

Let's proceed step by step.

Step 1: Determine whether patients meeting the criterion can be identified based on the OMOP CDM—that is, whether the criterion is computable. You may consider reversing the expression or constructing a medical concept repository for mapping.

- You are retrospectively reviewing an OMOP database; therefore, events occurring after screening are not computable.

Now, proceed to analyze the given criterion.
""",
"""
{history}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
<|im_start|>user

**Step 2: Decompose the criterion into individual traits.**

- **Ensure each trait contains only one main entity.**
- **Include all meaningful information, including modifiers and specific details, in the traits.**
- **Exclude traits that do not narrow or specify the scope of the population (e.g., "men or women" or "any time").**
- When traits have disjunctive modifiers or constraints, separate them into individual traits.
- When constraints or modifier could be consider as another main entity, separate them into individual traits.

- When a main entity has conjunctive constraints, keep them in a single trait.
- The combination of all traits, maintaining their logical relationships, should represent the original criterion.
- Be mindful of the scope of modifiers or constraints.
- Do not treat examples (e.g.) in the criterion as the complete definition of the entity; treat each example and the original definition as separate traits.
- If the constraint is a range, do not generate traits for each value within the range.
- Do not do any math calculation in the trait.

**Example:**

*Given the criterion:*  
"Patients are taking A or B at a dose of C for D months."

**Decomposition into traits:**

- **1:** "Patients are taking A at a dose of C for D months."
- **2:** "Patients are taking B at a dose of C for D months."

*Note:* Each trait contains only one main entity.
---

**Another Example:**

*Given the criterion:*  
"The following medications are allowed if stable for >1 month: A, B, and C."

**Decomposition into traits:**

- **1:** "A is allowed if stable for >1 month."
- **2:** "B is allowed if stable for >1 month."
- **3:** "C is allowed if stable for >1 month."

*Note:* The constraint ">1 month" apply to A, B and C.

---

Now, proceed to decompose the given criterion into traits, ensuring each trait has only one main entity.
""", """
{history}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
<|im_start|>user

Step 3: Describe the logical relationships among the traits.

- Use 'AND' or 'OR' to represent relationships.
- Do not use 'NOT'; keep negations within the traits.
- Use grouping operators if necessary.
""", """
{history}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
<|im_start|>user

**Step 4: Verify that all relevant information from the original criterion is retained**

- Compare the traits you've generated with the original criterion.
- **Ensure that no meaningful information, modifiers, or constraints have been omitted.**
- **If any information is missing, revisit Step 1 to include it.**
- Be particularly attentive to modifiers and specific details that add context to the main entities.

---

Now, perform this verification for the given criterion.

""", """
{history}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
<|im_start|>user

**Step 5: Validate your results and organize your answers according to the given format.**

- Ensure each trait is sufficient for one SQL query but not overly complex.
- Combined traits should accurately represent the original criterion.
- Be cautious of overlapping or redundant traits.
- Make sure the logical relationships preserve the original intent.
- Confirm that all components are accurately identified and appropriately classified.

Please provide your final answer.

{format_instructions}

Respond with a plain JSON instance without additional explanations. Use lowercase for boolean values ('true', 'false') and double quotation marks. Indexing starts from 1.
"""]
criterion_prompts_w_system = [True, True, True, True, True]
trait_prompts = ["""
Given this trait: 
{og_text}

Your tasks are:

1. Determine if the trait is computable using the OMOP CDM.
2. Check for negation and rephrase to a positive form if necessary.
3. Identify the key components of the trait.
4. Validate and summarize the result.

Let's proceed step by step.

**Step 1: Determine computability**

- Assess if patients meeting the trait can be identified in the OMOP CDM.
- You may consider constructing a medical concept repository for mapping concepts.
""", """
{history}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
<|im_start|>user

**Step 2: Check for negation and rephrase**

- Identify negative words ('no', 'not', 'without', 'absence of', etc.).
- If negation is present, rephrase the trait to describe the opposite population in positive terms.
- Do not consider standardized terms with negative words as negations (e.g., 'non-Hodgkin lymphoma', 'unstable cardiac disease').

**Example:**

*Given the trait:*  
"Patients with no history of diabetes."

- **Negation Present:** Yes (word 'no')
- **Rephrased Trait:** "Patients with a history of diabetes."

*Note:* The rephrased trait describes the opposite population in positive terms.

---

Now, proceed to analyze the given trait:
""", """
{history}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
<|im_start|>user

**Step 3: Identify key components of the rephrased trait**

- **Main Entity:** Use original words from the trait to identify the main entity.
  - Main entities are standardized clinical terms found in medical terminology dictionaries (e.g., ICD, CPT, LOINC, SNOMED CT).
- **Main Entity Type:** Select from 'Condition', 'Measurement', 'Procedure', 'Drug', 'Observation', 'Visit', 'Demographic', 'Device', 'Other'.
- **Modifier:** Identify any words that add specificity or complexity to the main entity.
- **Constraint Detail:** Parse and represent constraints precisely for SQL query conditions.
  - Use symbols for constraints when applicable (e.g., '>', '<', '>=', '<=').
  - For numeric ranges, convert the range into a list of boundary conditions using inequality symbols.
- **Constraint Type:** Select from 'Value', 'Time', 'Duration time', 'Count', 'Form', 'Dosage', 'Route', 'Age', 'Race', 'Gender', 'Ethnicity', 'Other', or null.

**Example:**

*Given the trait:*
"Patients are taking A at a dose of C for D months."

- **Main Entity:** "A"
- **Main Entity Type:** "Drug"
- **Modifier:** None
- **Constraint Detail:** [C, D]
- **Constraint Type:** ["Dosage", "Duration time"]

---

*Another example:*

*Given the trait:*
"Current, severe psychopathology."

- **Main Entity:** "psychopathology"
- **Main Entity Type:** "Condition"
- **Modifier:** "Severe"
- **Constraint Detail:** ["Current"]
- **Constraint Type:** ["Time"]

---

*Important Notes:*

- Separate Modifier and Constraint Details from the Main Entity if they prevent direct linkage to medical terminology dictionaries.
- Ensure there is no overlap between the Main Entity, Modifier, and Constraint Detail.
- Use original words from the trait to identify the main entity.
- Main entities are standardized clinical terms that can be found in medical terminology dictionaries such as ICD, CPT, LOINC, and SNOMED CT
- The number of Constraint Types matches the number of Constraint Details
""", """
{history}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
<|im_start|>user

**Step 4: Validate and summarize the result**

{format_instructions}

- Set 'negation' to true if the trait was rephrased from a negative form; otherwise, false.
- Remove information that does not influence the scope (e.g., 'any origin').
- Ensure the negation, main entity, modifier, and constraint detail correctly represent the trait and there is no overlap.
- Ensure the constraint detail contains all the meaningful constraints in the trait
- Confirm that all components are accurately identified and appropriately classified.

Provide your final answer in the specified format without any explanation.
"""]
trait_prompts_w_system = [True, True, True, True]