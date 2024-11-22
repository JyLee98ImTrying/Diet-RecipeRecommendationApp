def format_recipe_instructions(instructions):
    """Format recipe instructions from c() format to numbered list."""
    if not isinstance(instructions, str):
        return []
    instructions = instructions.replace('c(', '').replace(')', '')
    steps = [step.strip().strip('"') for step in instructions.split('",')]
    return steps

def combine_ingredients(quantities, parts):
    """Combine ingredient quantities and parts into natural language format."""
    if pd.isna(quantities) or pd.isna(parts):
        return []
        
    try:
        def parse_r_vector(text):
            if not isinstance(text, str):
                return []
            text = text.replace('c(', '').replace(')', '')
            items = text.split(',')
            cleaned = []
            for item in items:
                item = item.strip().strip('"').strip("'")
                if item.upper() != 'NA':
                    cleaned.append(item)
            return cleaned

        quantities_list = parse_r_vector(quantities)
        parts_list = parse_r_vector(parts)
        
        ingredients = []
        for i in range(len(parts_list)):
            if i < len(quantities_list) and quantities_list[i] and quantities_list[i].upper() != 'NA':
                ingredients.append(f"{quantities_list[i]} {parts_list[i]}".strip())
            else:
                ingredients.append(parts_list[i])
        
        return [ing for ing in ingredients if ing]
        
    except Exception as e:
        st.error(f"Error processing ingredients: {str(e)}")
        return []

def calculate_caloric_needs(gender, weight, height, age):
    if gender == "Female":
        BMR = 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)
    else:
        BMR = 66 + (13.7 * weight) + (5 * height) - (6.8 * age)
    return BMR
