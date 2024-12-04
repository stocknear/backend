with open("json/stock-screener/data.json", 'rb') as file:
    try:
        data = file.read()
        print(data[14807230:14807250])  # Print the problematic section
    except Exception as e:
        print(f"Error reading file: {e}")