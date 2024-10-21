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



revenue_sources = [{'name': 'Product', 'value': '17080000000', 'date': '2024-03-31'}, {'name': 'ServiceOther', 'value': '44778000000', 'date': '2024-03-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '24832000000', 'date': '2024-03-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '13911000000', 'date': '2024-03-31'}, {'name': 'Windows', 'value': '5929000000', 'date': '2024-03-31'}, {'name': 'Gaming', 'value': '5451000000', 'date': '2024-03-31'}, {'name': 'LinkedInCorporation', 'value': '4013000000', 'date': '2024-03-31'}, {'name': 'SearchAndNewsAdvertising', 'value': '3134000000', 'date': '2024-03-31'}, {'name': 'EnterpriseAndPartnerServices', 'value': '1861000000', 'date': '2024-03-31'}, {'name': 'DynamicsProductsAndCloudServices', 'value': '1646000000', 'date': '2024-03-31'}, {'name': 'Devices', 'value': '1067000000', 'date': '2024-03-31'}, {'name': 'OtherProductsAndServices', 'value': '14000000', 'date': '2024-03-31'}, {'name': 'Product', 'value': '18941000000', 'date': '2023-12-31'}, {'name': 'ServiceOther', 'value': '43079000000', 'date': '2023-12-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '23953000000', 'date': '2023-12-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '13477000000', 'date': '2023-12-31'}, {'name': 'Gaming', 'value': '7111000000', 'date': '2023-12-31'}, {'name': 'Windows', 'value': '5262000000', 'date': '2023-12-31'}, {'name': 'LinkedInCorporation', 'value': '4195000000', 'date': '2023-12-31'}, {'name': 'SearchAndNewsAdvertising', 'value': '3220000000', 'date': '2023-12-31'}, {'name': 'EnterpriseAndPartnerServices', 'value': '1917000000', 'date': '2023-12-31'}, {'name': 'DynamicsProductsAndCloudServices', 'value': '1576000000', 'date': '2023-12-31'}, {'name': 'Devices', 'value': '1298000000', 'date': '2023-12-31'}, {'name': 'OtherProductsAndServices', 'value': '11000000', 'date': '2023-12-31'}, {'name': 'Product', 'value': '15535000000', 'date': '2023-09-30'}, {'name': 'ServiceOther', 'value': '40982000000', 'date': '2023-09-30'}, {'name': 'ServerProductsAndCloudServices', 'value': '22308000000', 'date': '2023-09-30'}, {'name': 'OfficeProductsAndCloudServices', 'value': '13140000000', 'date': '2023-09-30'}, {'name': 'Windows', 'value': '5567000000', 'date': '2023-09-30'}, {'name': 'Gaming', 'value': '3919000000', 'date': '2023-09-30'}, {'name': 'LinkedInCorporation', 'value': '3913000000', 'date': '2023-09-30'}, {'name': 'SearchAndNewsAdvertising', 'value': '3053000000', 'date': '2023-09-30'}, {'name': 'EnterpriseAndPartnerServices', 'value': '1944000000', 'date': '2023-09-30'}, {'name': 'Dynamics', 'value': '1540000000', 'date': '2023-09-30'}, {'name': 'Devices', 'value': '1125000000', 'date': '2023-09-30'}, {'name': 'OtherProductsAndServices', 'value': '8000000', 'date': '2023-09-30'}, {'name': 'Product', 'value': '15588000000', 'date': '2023-03-31'}, {'name': 'ServiceOther', 'value': '37269000000', 'date': '2023-03-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '20025000000', 'date': '2023-03-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '12438000000', 'date': '2023-03-31'}, {'name': 'Windows', 'value': '5328000000', 'date': '2023-03-31'}, {'name': 'Gaming', 'value': '3607000000', 'date': '2023-03-31'}, {'name': 'LinkedInCorporation', 'value': '3697000000', 'date': '2023-03-31'}, {'name': 'SearchAndNewsAdvertising', 'value': '3045000000', 'date': '2023-03-31'}, {'name': 'EnterpriseServices', 'value': '2007000000', 'date': '2023-03-31'}, {'name': 'Devices', 'value': '1282000000', 'date': '2023-03-31'}, {'name': 'OtherProductsAndServices', 'value': '1428000000', 'date': '2023-03-31'}, {'name': 'Product', 'value': '16517000000', 'date': '2022-12-31'}, {'name': 'ServiceOther', 'value': '36230000000', 'date': '2022-12-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '19594000000', 'date': '2022-12-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '11837000000', 'date': '2022-12-31'}, {'name': 'Windows', 'value': '4808000000', 'date': '2022-12-31'}, {'name': 'Gaming', 'value': '4758000000', 'date': '2022-12-31'}, {'name': 'LinkedInCorporation', 'value': '3876000000', 'date': '2022-12-31'}, {'name': 'SearchAndNewsAdvertising', 'value': '3223000000', 'date': '2022-12-31'}, {'name': 'EnterpriseServices', 'value': '1862000000', 'date': '2022-12-31'}, {'name': 'Devices', 'value': '1430000000', 'date': '2022-12-31'}, {'name': 'OtherProductsAndServices', 'value': '1359000000', 'date': '2022-12-31'}, {'name': 'Product', 'value': '15741000000', 'date': '2022-09-30'}, {'name': 'ServiceOther', 'value': '34381000000', 'date': '2022-09-30'}, {'name': 'ServerProductsAndCloudServices', 'value': '18388000000', 'date': '2022-09-30'}, {'name': 'OfficeProductsAndCloudServices', 'value': '11548000000', 'date': '2022-09-30'}, {'name': 'Windows', 'value': '5313000000', 'date': '2022-09-30'}, {'name': 'LinkedInCorporation', 'value': '3663000000', 'date': '2022-09-30'}, {'name': 'Gaming', 'value': '3610000000', 'date': '2022-09-30'}, {'name': 'SearchAndNewsAdvertising', 'value': '2928000000', 'date': '2022-09-30'}, {'name': 'EnterpriseServices', 'value': '1876000000', 'date': '2022-09-30'}, {'name': 'Devices', 'value': '1448000000', 'date': '2022-09-30'}, {'name': 'OtherProductsAndServices', 'value': '1348000000', 'date': '2022-09-30'}, {'name': 'Product', 'value': '17366000000', 'date': '2022-03-31'}, {'name': 'ServiceOther', 'value': '31994000000', 'date': '2022-03-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '17038000000', 'date': '2022-03-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '11164000000', 'date': '2022-03-31'}, {'name': 'Windows', 'value': '6077000000', 'date': '2022-03-31'}, {'name': 'Gaming', 'value': '3740000000', 'date': '2022-03-31'}, {'name': 'LinkedInCorporation', 'value': '3437000000', 'date': '2022-03-31'}, {'name': 'SearchAndNewsAdvertising', 'value': '2945000000', 'date': '2022-03-31'}, {'name': 'EnterpriseServices', 'value': '1891000000', 'date': '2022-03-31'}, {'name': 'Devices', 'value': '1764000000', 'date': '2022-03-31'}, {'name': 'OtherProductsAndServices', 'value': '1304000000', 'date': '2022-03-31'}, {'name': 'Product', 'value': '20779000000', 'date': '2021-12-31'}, {'name': 'ServiceOther', 'value': '30949000000', 'date': '2021-12-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '16375000000', 'date': '2021-12-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '11251000000', 'date': '2021-12-31'}, {'name': 'Windows', 'value': '6600000000', 'date': '2021-12-31'}, {'name': 'Gaming', 'value': '5442000000', 'date': '2021-12-31'}, {'name': 'LinkedInCorporation', 'value': '3531000000', 'date': '2021-12-31'}, {'name': 'SearchAndNewsAdvertising', 'value': '3064000000', 'date': '2021-12-31'}, {'name': 'Devices', 'value': '2285000000', 'date': '2021-12-31'}, {'name': 'EnterpriseServices', 'value': '1823000000', 'date': '2021-12-31'}, {'name': 'OtherProductsAndServices', 'value': '1357000000', 'date': '2021-12-31'}, {'name': 'Product', 'value': '16631000000', 'date': '2021-09-30'}, {'name': 'ServiceOther', 'value': '28686000000', 'date': '2021-09-30'}, {'name': 'ServerProductsAndCloudServices', 'value': '15069000000', 'date': '2021-09-30'}, {'name': 'OfficeProductsAndCloudServices', 'value': '10808000000', 'date': '2021-09-30'}, {'name': 'Windows', 'value': '5676000000', 'date': '2021-09-30'}, {'name': 'Gaming', 'value': '3593000000', 'date': '2021-09-30'}, {'name': 'LinkedInCorporation', 'value': '3136000000', 'date': '2021-09-30'}, {'name': 'SearchAndNewsAdvertising', 'value': '2656000000', 'date': '2021-09-30'}, {'name': 'EnterpriseServices', 'value': '1791000000', 'date': '2021-09-30'}, {'name': 'Devices', 'value': '1361000000', 'date': '2021-09-30'}, {'name': 'OtherProductsAndServices', 'value': '1227000000', 'date': '2021-09-30'}, {'name': 'Product', 'value': '16873000000', 'date': '2021-03-31'}, {'name': 'ServiceOther', 'value': '24833000000', 'date': '2021-03-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '13204000000', 'date': '2021-03-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '10016000000', 'date': '2021-03-31'}, {'name': 'Windows', 'value': '5646000000', 'date': '2021-03-31'}, {'name': 'Gaming', 'value': '3533000000', 'date': '2021-03-31'}, {'name': 'LinkedInCorporation', 'value': '2562000000', 'date': '2021-03-31'}, {'name': 'SearchAdvertising', 'value': '2218000000', 'date': '2021-03-31'}, {'name': 'Devices', 'value': '1599000000', 'date': '2021-03-31'}, {'name': 'EnterpriseServices', 'value': '1803000000', 'date': '2021-03-31'}, {'name': 'OtherProductsAndServices', 'value': '1125000000', 'date': '2021-03-31'}, {'name': 'Product', 'value': '19460000000', 'date': '2020-12-31'}, {'name': 'ServiceOther', 'value': '23616000000', 'date': '2020-12-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '12729000000', 'date': '2020-12-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '9881000000', 'date': '2020-12-31'}, {'name': 'Windows', 'value': '5716000000', 'date': '2020-12-31'}, {'name': 'Gaming', 'value': '5031000000', 'date': '2020-12-31'}, {'name': 'LinkedInCorporation', 'value': '2577000000', 'date': '2020-12-31'}, {'name': 'SearchAdvertising', 'value': '2184000000', 'date': '2020-12-31'}, {'name': 'Devices', 'value': '2120000000', 'date': '2020-12-31'}, {'name': 'EnterpriseServices', 'value': '1695000000', 'date': '2020-12-31'}, {'name': 'OtherProductsAndServices', 'value': '1143000000', 'date': '2020-12-31'}, {'name': 'Product', 'value': '15803000000', 'date': '2020-09-30'}, {'name': 'ServiceOther', 'value': '21351000000', 'date': '2020-09-30'}, {'name': 'ServerProductsAndCloudServices', 'value': '11195000000', 'date': '2020-09-30'}, {'name': 'OfficeProductsAndCloudServices', 'value': '9278000000', 'date': '2020-09-30'}, {'name': 'Windows', 'value': '5305000000', 'date': '2020-09-30'}, {'name': 'Gaming', 'value': '3092000000', 'date': '2020-09-30'}, {'name': 'LinkedInCorporation', 'value': '2206000000', 'date': '2020-09-30'}, {'name': 'SearchAdvertising', 'value': '1789000000', 'date': '2020-09-30'}, {'name': 'EnterpriseServices', 'value': '1637000000', 'date': '2020-09-30'}, {'name': 'Devices', 'value': '1620000000', 'date': '2020-09-30'}, {'name': 'OtherProductsAndServices', 'value': '1032000000', 'date': '2020-09-30'}, {'name': 'Product', 'value': '15871000000', 'date': '2020-03-31'}, {'name': 'ServiceOther', 'value': '19150000000', 'date': '2020-03-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '10490000000', 'date': '2020-03-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '8920000000', 'date': '2020-03-31'}, {'name': 'Windows', 'value': '5220000000', 'date': '2020-03-31'}, {'name': 'Gaming', 'value': '2349000000', 'date': '2020-03-31'}, {'name': 'SearchAdvertising', 'value': '1986000000', 'date': '2020-03-31'}, {'name': 'LinkedInCorporation', 'value': '2050000000', 'date': '2020-03-31'}, {'name': 'EnterpriseServices', 'value': '1633000000', 'date': '2020-03-31'}, {'name': 'Devices', 'value': '1412000000', 'date': '2020-03-31'}, {'name': 'OtherProductsAndServices', 'value': '961000000', 'date': '2020-03-31'}, {'name': 'Product', 'value': '18255000000', 'date': '2019-12-31'}, {'name': 'ServiceOther', 'value': '18651000000', 'date': '2019-12-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '10119000000', 'date': '2019-12-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '8983000000', 'date': '2019-12-31'}, {'name': 'Windows', 'value': '5593000000', 'date': '2019-12-31'}, {'name': 'Gaming', 'value': '3327000000', 'date': '2019-12-31'}, {'name': 'SearchAdvertising', 'value': '2163000000', 'date': '2019-12-31'}, {'name': 'LinkedInCorporation', 'value': '2102000000', 'date': '2019-12-31'}, {'name': 'Devices', 'value': '2048000000', 'date': '2019-12-31'}, {'name': 'EnterpriseServices', 'value': '1612000000', 'date': '2019-12-31'}, {'name': 'OtherProductsAndServices', 'value': '959000000', 'date': '2019-12-31'}, {'name': 'Product', 'value': '15768000000', 'date': '2019-09-30'}, {'name': 'ServiceOther', 'value': '17287000000', 'date': '2019-09-30'}, {'name': 'ServerProductsAndCloudServices', 'value': '9192000000', 'date': '2019-09-30'}, {'name': 'OfficeProductsAndCloudServices', 'value': '8466000000', 'date': '2019-09-30'}, {'name': 'Windows', 'value': '5353000000', 'date': '2019-09-30'}, {'name': 'Gaming', 'value': '2542000000', 'date': '2019-09-30'}, {'name': 'SearchAdvertising', 'value': '1991000000', 'date': '2019-09-30'}, {'name': 'LinkedInCorporation', 'value': '1909000000', 'date': '2019-09-30'}, {'name': 'EnterpriseServices', 'value': '1545000000', 'date': '2019-09-30'}, {'name': 'Devices', 'value': '1202000000', 'date': '2019-09-30'}, {'name': 'OtherProductsAndServices', 'value': '855000000', 'date': '2019-09-30'}, {'name': 'Product', 'value': '15448000000', 'date': '2019-03-31'}, {'name': 'ServiceOther', 'value': '15123000000', 'date': '2019-03-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '7889000000', 'date': '2019-03-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '8053000000', 'date': '2019-03-31'}, {'name': 'Windows', 'value': '4944000000', 'date': '2019-03-31'}, {'name': 'Gaming', 'value': '2363000000', 'date': '2019-03-31'}, {'name': 'SearchAdvertising', 'value': '1911000000', 'date': '2019-03-31'}, {'name': 'LinkedInCorporation', 'value': '1696000000', 'date': '2019-03-31'}, {'name': 'Devices', 'value': '1423000000', 'date': '2019-03-31'}, {'name': 'EnterpriseServices', 'value': '1542000000', 'date': '2019-03-31'}, {'name': 'OtherProductsAndServices', 'value': '750000000', 'date': '2019-03-31'}, {'name': 'Product', 'value': '16219000000', 'date': '2018-12-31'}, {'name': 'ServiceOther', 'value': '16252000000', 'date': '2018-12-31'}, {'name': 'OfficeProductsAndCloudServices', 'value': '7747000000', 'date': '2018-12-31'}, {'name': 'ServerProductsAndCloudServices', 'value': '7791000000', 'date': '2018-12-31'}, {'name': 'Windows', 'value': '4758000000', 'date': '2018-12-31'}, {'name': 'Gaming', 'value': '4232000000', 'date': '2018-12-31'}, {'name': 'SearchAdvertising', 'value': '1976000000', 'date': '2018-12-31'}, {'name': 'LinkedInCorporation', 'value': '1693000000', 'date': '2018-12-31'}, {'name': 'Devices', 'value': '1948000000', 'date': '2018-12-31'}, {'name': 'EnterpriseServices', 'value': '1521000000', 'date': '2018-12-31'}, {'name': 'OtherProductsAndServices', 'value': '805000000', 'date': '2018-12-31'}]


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
    }

    # Filter out unwanted categories
    excluded_names = {'automotiveleasing ','officeproductsandcloudservices','serverproductsandcloudservices','automotiverevenues','automotive','computeandnetworking','graphics','gpu','automotivesegment','energygenerationandstoragesales','energygenerationandstorage','automotivesaleswithoutresalevalueguarantee','salesandservices','compute', 'networking', 'cloudserviceagreements', 'digital', 'allother', 'preownedvideogameproducts'}
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

    final_result = {'names': top_names, 'history': res_list}

    #print(len(top_names), len(res_list[4]['value']))
    print(res_list)
    return final_result


def generate_geography_dataset(dataset):

    country_replacements = {
        "country:us": "United States",
        "country:cn": "China",
        "chinaincludinghongkong": "China"
    }

    # Custom order for specific countries
    custom_order = {
        'United States': 2,
        'China': 1,
        'Other': 0
    }

    for item in dataset:
        item['date'] = closest_quarter_end(item['date'])
        name = item.get('name').lower()
        value = int(float(item.get('value')))
        if name in country_replacements:
            item['name'] = country_replacements[name]
            item['value'] = value
        else:
            item['name'] = 'Other'
            item['value'] = value

    dataset = aggregate_other_values(dataset)
    dataset = sorted(
        dataset,
        key=lambda item: (datetime.strptime(item['date'], '%Y-%m-%d'), custom_order.get(item['name'], 3)),
        reverse = True
    )

    dataset = compute_q4_results(dataset)
    result = {}

    # Iterate through the original data
    for item in dataset:
        # Get the date and value
        date = item['date']
        value = item['value']
        
        # Initialize the dictionary for the date if not already done
        if date not in result:
            result[date] = {'date': date, 'value': []}
        
        # Append the value to the list
        result[date]['value'].append(value)

    # Convert the result dictionary to a list
    # Convert the result dictionary to a list
    final_result = list(result.values())

    # Print the final result
    final_result = add_value_growth(final_result)
    print(final_result)


generate_revenue_dataset(revenue_sources)


