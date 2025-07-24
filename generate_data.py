import itertools
import pandas as pd
import argparse

# Define the attributes and their values
attributes = {
    "attractive": ["attractive", "no_attractive"],
    "male": ["man", "woman"],
    "age": ["child", "young", "old"],
    "color_hair": ["black_hair", "blonde_hair", "brown_hair", "gray_hair"],
    "hair_type": ["bald", "straight_hair", "wavy_hair"],
    "skin": ["black_skin", "white_skin"],
    "color_eyes": ["black_eyes", "blue_eyes", "green_eyes"],
    "Mustach_beard": ["Mustach", "beard", "Mustach_Beard"],
    "makeup": ["No_makeup", "makeup", "heavy_makeup"],
    "Nose": ["pointy_nose", "big_nose"],
    "Face": ["oval_face", "round_face", "square_face"],
}

def convert_to_one_hot(row, attributes):
    one_hot_row = []
    for attr, values in attributes.items():
        one_hot = [0] * len(values)
        one_hot[row[attr]] = 1
        one_hot_row.extend(one_hot)
    return one_hot_row

def generate_description(row):
    words = []
    attr = ''
    if row.get('attractive') == 1:
        attr = 'attractive'
    elif row.get('no_attractive') == 1:
        attr = 'no attractive'

    gender = ''
    if row.get('man') == 1:
        gender = 'man'
    elif row.get('woman') == 1:
        gender = 'woman'

    if attr:
        words.append(attr)
    if gender:
        words.append(gender)

    skip_cols = {'attractive', 'no_attractive', 'man', 'woman'}
    other_attrs = [col.replace('_', ' ') for col in row.index if row[col] == 1 and col not in skip_cols]
    if other_attrs:
        words.extend(['and ' + attr for attr in other_attrs])

    return ' '.join(words)

def main(output_path):
    # Generate all combinations
    attribute_ranges = [range(len(values)) for values in attributes.values()]
    combinations = list(itertools.product(*attribute_ranges))
    columns = list(attributes.keys())
    df = pd.DataFrame(combinations, columns=columns)

    # One-hot encoding
    one_hot_data = [convert_to_one_hot(row, attributes) for _, row in df.iterrows()]
    one_hot_columns = [val for values in attributes.values() for val in values]
    df_one_hot = pd.DataFrame(one_hot_data, columns=one_hot_columns)

    # Generate description
    df_one_hot['description'] = df_one_hot.apply(generate_description, axis=1)

    # Save to CSV
    df_one_hot.to_csv(output_path, index=False)
    print(f"✅ File saved successfully to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one-hot encoded attribute combinations and descriptions.")
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to save the generated CSV file (e.g., output/df_input_description.csv)"
    )
    args = parser.parse_args()
    main(args.output)
