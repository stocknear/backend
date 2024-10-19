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

# Define quarter-end dates for a given year
def closest_quarter_end(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    
    # Define quarter end dates for the year
    q1 = datetime(year, 3, 31)
    q2 = datetime(year, 6, 30)
    q3 = datetime(year, 9, 30)
    q4 = datetime(year, 12, 31)
    
    # Find the closest quarter date
    closest = min([q1, q2, q3, q4], key=lambda d: abs(d - date))
    
    # Return the closest quarter date in 'YYYY-MM-DD' format
    return closest.strftime("%Y-%m-%d")

revenue_sources = [{'name': 'DataCenter', 'value': '26272000000', 'date': '2024-07-28'}, {'name': 'Compute', 'value': '22604000000', 'date': '2024-07-28'}, {'name': 'Networking', 'value': '3668000000', 'date': '2024-07-28'}, {'name': 'Gaming', 'value': '2880000000', 'date': '2024-07-28'}, {'name': 'ProfessionalVisualization', 'value': '454000000', 'date': '2024-07-28'}, {'name': 'Automotive', 'value': '346000000', 'date': '2024-07-28'}, {'name': 'OEMAndOther', 'value': '88000000', 'date': '2024-07-28'}, {'name': 'DataCenter', 'value': '22563000000', 'date': '2024-04-28'}, {'name': 'Compute', 'value': '19392000000', 'date': '2024-04-28'}, {'name': 'Networking', 'value': '3171000000', 'date': '2024-04-28'}, {'name': 'Gaming', 'value': '2647000000', 'date': '2024-04-28'}, {'name': 'ProfessionalVisualization', 'value': '427000000', 'date': '2024-04-28'}, {'name': 'Automotive', 'value': '329000000', 'date': '2024-04-28'}, {'name': 'OEMAndOther', 'value': '78000000', 'date': '2024-04-28'}, {'name': 'DataCenter', 'value': '14514000000', 'date': '2023-10-29'}, {'name': 'Gaming', 'value': '2856000000', 'date': '2023-10-29'}, {'name': 'ProfessionalVisualization', 'value': '416000000', 'date': '2023-10-29'}, {'name': 'Automotive', 'value': '261000000', 'date': '2023-10-29'}, {'name': 'OEMAndOther', 'value': '73000000', 'date': '2023-10-29'}, {'name': 'DataCenter', 'value': '10323000000', 'date': '2023-07-30'}, {'name': 'Gaming', 'value': '2486000000', 'date': '2023-07-30'}, {'name': 'ProfessionalVisualization', 'value': '379000000', 'date': '2023-07-30'}, {'name': 'Automotive', 'value': '253000000', 'date': '2023-07-30'}, {'name': 'OEMAndOther', 'value': '66000000', 'date': '2023-07-30'}, {'name': 'CloudServiceAgreements', 'value': '2430000000', 'date': '2023-04-30'}, {'name': 'DataCenter', 'value': '4284000000', 'date': '2023-04-30'}, {'name': 'Gaming', 'value': '2240000000', 'date': '2023-04-30'}, {'name': 'ProfessionalVisualization', 'value': '295000000', 'date': '2023-04-30'}, {'name': 'Automotive', 'value': '296000000', 'date': '2023-04-30'}, {'name': 'OEMAndOther', 'value': '77000000', 'date': '2023-04-30'}, {'name': 'DataCenter', 'value': '3833000000', 'date': '2022-10-30'}, {'name': 'Gaming', 'value': '1574000000', 'date': '2022-10-30'}, {'name': 'ProfessionalVisualization', 'value': '200000000', 'date': '2022-10-30'}, {'name': 'Automotive', 'value': '251000000', 'date': '2022-10-30'}, {'name': 'OEMAndOther', 'value': '73000000', 'date': '2022-10-30'}, {'name': 'Gaming', 'value': '2042000000', 'date': '2022-07-31'}, {'name': 'DataCenter', 'value': '3806000000', 'date': '2022-07-31'}, {'name': 'ProfessionalVisualization', 'value': '496000000', 'date': '2022-07-31'}, {'name': 'Automotive', 'value': '220000000', 'date': '2022-07-31'}, {'name': 'OEMAndOther', 'value': '140000000', 'date': '2022-07-31'}, {'name': 'Gaming', 'value': '3620000000', 'date': '2022-05-01'}, {'name': 'DataCenter', 'value': '3750000000', 'date': '2022-05-01'}, {'name': 'ProfessionalVisualization', 'value': '622000000', 'date': '2022-05-01'}, {'name': 'Automotive', 'value': '138000000', 'date': '2022-05-01'}, {'name': 'OEMAndOther', 'value': '158000000', 'date': '2022-05-01'}, {'name': 'Gaming', 'value': '3221000000', 'date': '2021-10-31'}, {'name': 'DataCenter', 'value': '2936000000', 'date': '2021-10-31'}, {'name': 'ProfessionalVisualization', 'value': '577000000', 'date': '2021-10-31'}, {'name': 'Automotive', 'value': '135000000', 'date': '2021-10-31'}, {'name': 'OEMAndOther', 'value': '234000000', 'date': '2021-10-31'}, {'name': 'Gaming', 'value': '3061000000', 'date': '2021-08-01'}, {'name': 'DataCenter', 'value': '2366000000', 'date': '2021-08-01'}, {'name': 'ProfessionalVisualization', 'value': '519000000', 'date': '2021-08-01'}, {'name': 'Automotive', 'value': '152000000', 'date': '2021-08-01'}, {'name': 'OEMAndOther', 'value': '409000000', 'date': '2021-08-01'}, {'name': 'Gaming', 'value': '2760000000', 'date': '2021-05-02'}, {'name': 'DataCenter', 'value': '2048000000', 'date': '2021-05-02'}, {'name': 'ProfessionalVisualization', 'value': '372000000', 'date': '2021-05-02'}, {'name': 'Automotive', 'value': '154000000', 'date': '2021-05-02'}, {'name': 'OEMAndOther', 'value': '327000000', 'date': '2021-05-02'}, {'name': 'Gaming', 'value': '2271000000', 'date': '2020-10-25'}, {'name': 'ProfessionalVisualization', 'value': '236000000', 'date': '2020-10-25'}, {'name': 'DataCenter', 'value': '1900000000', 'date': '2020-10-25'}, {'name': 'Automotive', 'value': '125000000', 'date': '2020-10-25'}, {'name': 'OEMAndOther', 'value': '194000000', 'date': '2020-10-25'}, {'name': 'Gaming', 'value': '1654000000', 'date': '2020-07-26'}, {'name': 'ProfessionalVisualization', 'value': '203000000', 'date': '2020-07-26'}, {'name': 'DataCenter', 'value': '1752000000', 'date': '2020-07-26'}, {'name': 'Automotive', 'value': '111000000', 'date': '2020-07-26'}, {'name': 'OEMAndOther', 'value': '146000000', 'date': '2020-07-26'}, {'name': 'Gaming', 'value': '1339000000', 'date': '2020-04-26'}, {'name': 'ProfessionalVisualization', 'value': '307000000', 'date': '2020-04-26'}, {'name': 'DataCenter', 'value': '1141000000', 'date': '2020-04-26'}, {'name': 'Automotive', 'value': '155000000', 'date': '2020-04-26'}, {'name': 'OEMAndOther', 'value': '138000000', 'date': '2020-04-26'}, {'name': 'Gaming', 'value': '1659000000', 'date': '2019-10-27'}, {'name': 'ProfessionalVisualization', 'value': '324000000', 'date': '2019-10-27'}, {'name': 'DataCenter', 'value': '726000000', 'date': '2019-10-27'}, {'name': 'Automotive', 'value': '162000000', 'date': '2019-10-27'}, {'name': 'OEMAndOther', 'value': '143000000', 'date': '2019-10-27'}, {'name': 'Gaming', 'value': '1313000000', 'date': '2019-07-28'}, {'name': 'ProfessionalVisualization', 'value': '291000000', 'date': '2019-07-28'}, {'name': 'Datacenter', 'value': '655000000', 'date': '2019-07-28'}, {'name': 'Automotive', 'value': '209000000', 'date': '2019-07-28'}, {'name': 'OEMIP', 'value': '111000000', 'date': '2019-07-28'}, {'name': 'Automotive', 'value': '145000000', 'date': '2018-04-29'}, {'name': 'Datacenter', 'value': '701000000', 'date': '2018-04-29'}, {'name': 'Gaming', 'value': '1723000000', 'date': '2018-04-29'}, {'name': 'OEMIP', 'value': '387000000', 'date': '2018-04-29'}, {'name': 'ProfessionalVisualization', 'value': '251000000', 'date': '2018-04-29'}]
geographic_sources = [{'name': 'country:US', 'value': '13022000000', 'date': '2024-07-28'}, {'name': 'country:TW', 'value': '5740000000', 'date': '2024-07-28'}, {'name': 'country:SG', 'value': '5622000000', 'date': '2024-07-28'}, {'name': 'ChinaIncludingHongKong', 'value': '3667000000', 'date': '2024-07-28'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '1989000000', 'date': '2024-07-28'}, {'name': 'country:US', 'value': '13496000000', 'date': '2024-04-28'}, {'name': 'country:TW', 'value': '4373000000', 'date': '2024-04-28'}, {'name': 'country:SG', 'value': '4037000000', 'date': '2024-04-28'}, {'name': 'ChinaIncludingHongKong', 'value': '2491000000', 'date': '2024-04-28'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '1647000000', 'date': '2024-04-28'}, {'name': 'country:US', 'value': '6302000000', 'date': '2023-10-29'}, {'name': 'country:TW', 'value': '4333000000', 'date': '2023-10-29'}, {'name': 'ChinaIncludingHongKong', 'value': '4030000000', 'date': '2023-10-29'}, {'name': 'country:SG', 'value': '2702000000', 'date': '2023-10-29'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '753000000', 'date': '2023-10-29'}, {'name': 'country:US', 'value': '6043000000', 'date': '2023-07-30'}, {'name': 'country:TW', 'value': '2839000000', 'date': '2023-07-30'}, {'name': 'ChinaIncludingHongKong', 'value': '2740000000', 'date': '2023-07-30'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '1885000000', 'date': '2023-07-30'}, {'name': 'country:US', 'value': '2385000000', 'date': '2023-04-30'}, {'name': 'country:TW', 'value': '1796000000', 'date': '2023-04-30'}, {'name': 'ChinaIncludingHongKong', 'value': '1590000000', 'date': '2023-04-30'}, {'name': 'country:SG', 'value': '762000000', 'date': '2023-04-30'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '659000000', 'date': '2023-04-30'}, {'name': 'country:US', 'value': '2148000000', 'date': '2022-10-30'}, {'name': 'country:TW', 'value': '1153000000', 'date': '2022-10-30'}, {'name': 'ChinaIncludingHongKong', 'value': '1148000000', 'date': '2022-10-30'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '1482000000', 'date': '2022-10-30'}, {'name': 'country:US', 'value': '1988000000', 'date': '2022-07-31'}, {'name': 'ChinaIncludingHongKong', 'value': '1602000000', 'date': '2022-07-31'}, {'name': 'country:TW', 'value': '1204000000', 'date': '2022-07-31'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '1910000000', 'date': '2022-07-31'}, {'name': 'country:TW', 'value': '2777000000', 'date': '2022-05-01'}, {'name': 'ChinaIncludingHongKong', 'value': '2081000000', 'date': '2022-05-01'}, {'name': 'country:US', 'value': '1932000000', 'date': '2022-05-01'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '1498000000', 'date': '2022-05-01'}, {'name': 'country:IL', 'value': '1700000000', 'date': '2021-10-31'}, {'name': 'country:GB', 'value': '231000000', 'date': '2021-10-31'}, {'name': 'country:TW', 'value': '2187000000', 'date': '2021-10-31'}, {'name': 'ChinaIncludingHongKong', 'value': '2017000000', 'date': '2021-10-31'}, {'name': 'OtherAsiaPacific', 'value': '1067000000', 'date': '2021-10-31'}, {'name': 'country:US', 'value': '1126000000', 'date': '2021-10-31'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '366000000', 'date': '2021-10-31'}, {'name': 'country:IL', 'value': '1600000000', 'date': '2021-08-01'}, {'name': 'country:GB', 'value': '231000000', 'date': '2021-08-01'}, {'name': 'country:TW', 'value': '1961000000', 'date': '2021-08-01'}, {'name': 'ChinaIncludingHongKong', 'value': '1720000000', 'date': '2021-08-01'}, {'name': 'OtherAsiaPacific', 'value': '1047000000', 'date': '2021-08-01'}, {'name': 'country:US', 'value': '996000000', 'date': '2021-08-01'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '354000000', 'date': '2021-08-01'}, {'name': 'country:TW', 'value': '1784000000', 'date': '2021-05-02'}, {'name': 'ChinaIncludingHongKong', 'value': '1391000000', 'date': '2021-05-02'}, {'name': 'OtherAsiaPacific', 'value': '1001000000', 'date': '2021-05-02'}, {'name': 'country:US', 'value': '768000000', 'date': '2021-05-02'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '336000000', 'date': '2021-05-02'}, {'name': 'country:TW', 'value': '1296000000', 'date': '2020-10-25'}, {'name': 'ChinaIncludingHongKong', 'value': '1113000000', 'date': '2020-10-25'}, {'name': 'OtherAsiaPacific', 'value': '955000000', 'date': '2020-10-25'}, {'name': 'country:US', 'value': '890000000', 'date': '2020-10-25'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '225000000', 'date': '2020-10-25'}, {'name': 'country:TW', 'value': '954000000', 'date': '2020-07-26'}, {'name': 'country:US', 'value': '944000000', 'date': '2020-07-26'}, {'name': 'ChinaIncludingHongKong', 'value': '855000000', 'date': '2020-07-26'}, {'name': 'OtherAsiaPacific', 'value': '698000000', 'date': '2020-07-26'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '175000000', 'date': '2020-07-26'}, {'name': 'country:TW', 'value': '813000000', 'date': '2020-04-26'}, {'name': 'ChinaIncludingHongKong', 'value': '758000000', 'date': '2020-04-26'}, {'name': 'OtherAsiaPacific', 'value': '607000000', 'date': '2020-04-26'}, {'name': 'country:US', 'value': '497000000', 'date': '2020-04-26'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '151000000', 'date': '2020-04-26'}, {'name': 'country:TW', 'value': '838000000', 'date': '2019-10-27'}, {'name': 'OtherAsiaPacific', 'value': '805000000', 'date': '2019-10-27'}, {'name': 'ChinaIncludingHongKong', 'value': '758000000', 'date': '2019-10-27'}, {'name': 'country:US', 'value': '236000000', 'date': '2019-10-27'}, {'name': 'AllOtherCountriesNotSeparatelyDisclosed', 'value': '161000000', 'date': '2019-10-27'}, {'name': 'OtherAsiaPacific', 'value': '756000000', 'date': '2019-07-28'}, {'name': 'country:TW', 'value': '635000000', 'date': '2019-07-28'}, {'name': 'country:CN', 'value': '583000000', 'date': '2019-07-28'}, {'name': 'country:US', 'value': '188000000', 'date': '2019-07-28'}, {'name': 'OtherAmericas', 'value': '129000000', 'date': '2019-07-28'}, {'name': 'country:CN', 'value': '754000000', 'date': '2018-04-29'}, {'name': 'country:TW', 'value': '967000000', 'date': '2018-04-29'}, {'name': 'country:US', 'value': '434000000', 'date': '2018-04-29'}, {'name': 'OtherAmericas', 'value': '234000000', 'date': '2018-04-29'}, {'name': 'OtherAsiaPacific', 'value': '583000000', 'date': '2018-04-29'}]


def generate_revenue_dataset(dataset):
    name_replacements = {
        "datacenter": "Data Center",
        "professionalvisualization": "Professional Visualization",
        "oemandother": "OEM and Other",
        "automotive": "Automotive and Robotics",
        "oemip": "OEM and Other",
        "gaming": "Gaming"
    }
    dataset = [revenue for revenue in dataset if revenue['name'] not in ['Compute', 'Networking']]


    for item in dataset:
        item['date'] = closest_quarter_end(item['date'])
        name = item.get('name').lower()
        value = int(item.get('value'))
        if name in name_replacements:
            item['name'] = name_replacements[name]
            item['value'] = int(value)

    # Custom order for specific countries
    custom_order = {
        'Data Center': 4,
        'Gaming': 3,
        'Professional Visualization': 2,
        'Automotive and Robotics': 1,
        'OEM and Other': 0
    }
    
    dataset = sorted(
        dataset,
        key=lambda item: (datetime.strptime(item['date'], '%Y-%m-%d'), custom_order.get(item['name'], 4)),
        reverse = True
    )

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
    final_result = list(result.values())

    # Print the final result
    final_result = add_value_growth(final_result)
    print(final_result)

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
        value = int(item.get('value'))
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
    final_result = list(result.values())

    # Print the final result
    print(final_result)


generate_revenue_dataset(revenue_sources)

#generate_geography_dataset(geographic_sources)

