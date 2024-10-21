from datetime import datetime
from collections import defaultdict


def add_value_growth(data):
    """
    Adds a new key 'valueGrowth' to each entry in the data list.

    Parameters:
    - data (list): A list of dictionaries containing date and value lists.

    Returns:
    - list: A new list with the 'valueGrowth' key added to each dictionary.
    """
    # Initialize a new list for the output data
    updated_data = []

    # Loop through the data from the latest to the oldest
    for i in range(len(data)):
        try:
            current_entry = data[i].copy()  # Create a copy of the current entry
            current_values = current_entry['value']

            # Initialize the growth percentages list
            if i < len(data) - 1:  # Only compute growth if there is a next entry
                next_values = data[i + 1]['value']
                growth_percentages = []

                for j in range(len(current_values)):
                    # Convert values to integers if they are strings
                    next_value = int(next_values[j]) if isinstance(next_values[j], (int, str)) else 0
                    current_value = int(current_values[j]) if isinstance(current_values[j], (int, str)) else 0
                    
                    # Calculate growth percentage if next_value is not zero
                    if next_value != 0:
                        growth = round(((current_value - next_value) / next_value) * 100,2)
                    else:
                        growth = None  # Cannot calculate growth if next value is zero

                    growth_percentages.append(growth)

                current_entry['valueGrowth'] = growth_percentages  # Add the growth percentages
            else:
                current_entry['valueGrowth'] = [None] * len(current_values)  # No growth for the last entry

            updated_data.append(current_entry)  # Append the updated entry to the output list
        except:
            pass

    return updated_data

def sort_by_latest_date_and_highest_value(data):
    # Define a key function to convert the date string to a datetime object
    # and use the negative of the integer value for descending order
    def sort_key(item):
        date = datetime.strptime(item['date'], '%Y-%m-%d')
        value = -int(item['value'])  # Negative for descending order
        return (date, value)
    
    # Sort the list
    sorted_data = sorted(data, key=sort_key, reverse=True)
    
    return sorted_data

def aggregate_other_values(data):
    aggregated = defaultdict(int)
    result = []

    # First pass: aggregate 'Other' values and keep non-'Other' items
    for item in data:
        date = item['date']
        value = int(item['value'])
        if item['name'] == 'Other':
            aggregated[date] += value
        else:
            result.append(item)

    # Second pass: add aggregated 'Other' values
    for date, value in aggregated.items():
        result.append({'name': 'Other', 'value': int(value), 'date': date})

    return sorted(result, key=lambda x: (x['date'], x['name']))




def generate_revenue_dataset(dataset):
    # Find all unique names and dates
    all_dates = sorted(set(item['date'] for item in dataset))
    all_names = sorted(set(item['name'] for item in dataset))
    
    # Check and fill missing combinations at the beginning
    name_date_map = defaultdict(lambda: defaultdict(lambda: None))
    for item in dataset:
        name_date_map[item['name']][item['date']] = item['value']
    
    # Ensure all names have entries for all dates
    for name in all_names:
        for date in all_dates:
            if date not in name_date_map[name]:
                dataset.append({'name': name, 'date': date, 'value': None})
    
    # Clean and process the dataset values
    processed_dataset = []
    for item in dataset:
        if item['value'] not in (None, '', 0):
            processed_dataset.append({
                'name': item['name'],
                'date': item['date'],
                'value': int(float(item['value']))
            })
        else:
            processed_dataset.append({
                'name': item['name'],
                'date': item['date'],
                'value': None
            })
        
    dataset = processed_dataset

    dataset = sorted(dataset, key=lambda item: datetime.strptime(item['date'], '%Y-%m-%d'), reverse=True)

    remember_names = set()  # Use a set for faster membership checks

    first_date = dataset[0]['date']

    # Iterate through dataset to remember names where date matches first_date and value is None
    for item in dataset:
        if item['date'] == first_date and item['value'] is None:
            remember_names.add(item['name'])

    # Use list comprehension to filter items not in remember_names
    dataset = [{**item} for item in dataset if item['name'] not in remember_names]



    name_replacements = {
        "datacenter": "Data Center",
        "professionalvisualization": "Visualization",
        "oemandother": "OEM & Other",
        "automotive": "Automotive",
        "oemip": "OEM & Other",
        "gaming": "Gaming",
        "mac": "Mac",
        "iphone": "IPhone",
        "ipad": "IPad",
        "wearableshomeandaccessories": "Wearables",
        "hardwareandaccessories": "Hardware & Accessories",
        "software": "Software",
        "collectibles": "Collectibles",
        "automotivesales": "Auto",
        "energygenerationandstoragesegment": "Energy and Storage",
        "servicesandother": "Services & Other",
        "automotiveregulatorycredits": "Regulatory Credits",
        "intelligentcloud": "Intelligent Cloud",
        "productivityandbusinessprocesses": "Productivity & Business",
        "searchandnewsadvertising": "Advertising",
        "linkedincorporation": "LinkedIn",
        "morepersonalcomputing": "More Personal Computing",
        "serviceother": "Service Other",
        "governmentoperatingsegment": "Government Operating"
    }

    # Filter out unwanted categories
    excluded_names = {'enterpriseembeddedandsemicustom','computingandgraphics','automotiveleasing ','officeproductsandcloudservices','serverproductsandcloudservices','automotiverevenues','automotive','computeandnetworking','graphics','gpu','automotivesegment','energygenerationandstoragesales','energygenerationandstorage','automotivesaleswithoutresalevalueguarantee','salesandservices','compute', 'networking', 'cloudserviceagreements', 'digital', 'allother', 'preownedvideogameproducts'}
    dataset = [revenue for revenue in dataset if revenue['name'].lower() not in excluded_names]

    # Process and clean the dataset
    for item in dataset:
        try:
            name = item.get('name').lower()
            value = int(float(item.get('value')))
            if name in name_replacements:
                item['name'] = name_replacements[name]
            item['value'] = value
        except:
            pass

    # Group by name and calculate total value
    name_totals = defaultdict(int)
    for item in dataset:
        name_totals[item['name']] += item['value'] if item['value'] != None else 0

    # Sort names by total value and get top 5, ensuring excluded names are not considered
    top_names = sorted(
        [(name, total) for name, total in name_totals.items() if name.lower() not in excluded_names],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    top_names = [name for name, _ in top_names]

    # Filter dataset to include only top 5 names
    dataset = [item for item in dataset if item['name'] in top_names]

    # Sort the dataset
    dataset.sort(key=lambda item: (datetime.strptime(item['date'], '%Y-%m-%d'), item['value'] if item['value'] != None else 0), reverse=True)



    
    # Process the data into the required format
    result = {}
    for item in dataset:
        date = item['date']
        value = item['value']
        if date not in result:
            result[date] = {'date': date, 'value': []}
        result[date]['value'].append(value)



    # Convert the result dictionary to a list
    res_list = list(result.values())
    
    # Add value growth (assuming add_value_growth function exists)
    res_list = add_value_growth(res_list)
    print(res_list)
    final_result = {'names': top_names, 'history': res_list}
    return final_result


revenue_sources = [{'name': 'GovernmentOperatingSegment', 'value': '370767000', 'date': '2024-06-30'}, {'name': 'Commercial', 'value': '307367000', 'date': '2024-06-30'}, {'name': 'GovernmentOperatingSegment', 'value': '335373000', 'date': '2024-03-31'}, {'name': 'Commercial', 'value': '298965000', 'date': '2024-03-31'}, {'name': 'GovernmentOperatingSegment', 'value': '307603000', 'date': '2023-09-30'}, {'name': 'Commercial', 'value': '250556000', 'date': '2023-09-30'}, {'name': 'GovernmentOperatingSegment', 'value': '301505000', 'date': '2023-06-30'}, {'name': 'Commercial', 'value': '231812000', 'date': '2023-06-30'}, {'name': 'GovernmentOperatingSegment', 'value': '289070000', 'date': '2023-03-31'}, {'name': 'Commercial', 'value': '236116000', 'date': '2023-03-31'}, {'name': 'GovernmentOperatingSegment', 'value': '273834000', 'date': '2022-09-30'}, {'name': 'Commercial', 'value': '204046000', 'date': '2022-09-30'}, {'name': 'Commercial', 'value': '210012000', 'date': '2022-06-30'}, {'name': 'Commercial', 'value': '204567000', 'date': '2022-03-31'}, {'name': 'LicenseAndService', 'value': '25000000', 'date': '2019-12-31'}, {'name': 'LicenseAndService', 'value': '25000000', 'date': '2020-09-30'}, {'name': 'Government', 'value': '162561000', 'date': '2020-09-30'}, {'name': 'Commercial', 'value': '126805000', 'date': '2020-09-30'}]

generate_revenue_dataset(revenue_sources)


