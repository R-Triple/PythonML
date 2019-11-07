# Machine Learning is programming to solve a problem not explicitly told to solve

# NumPy is a library with a very efficient and linear algebra functions.

# scikit-learn

# pandas makes it easy to load and work with large data sets like a spreadsheet on excel

def estimate_house_sales_price(num_of_bedrooms, sqft, neighborhood):
    price = 0
    #calculate price for number of bedrooms
    price += num_of_bedrooms * .841
    #calculate price for sqft
    price += sqft * 1.0
    #calulate price for how neighborhood affects
    price += neighborhood * 1.0
    # and finally, just a little extra salt for good measure
    price += 1.0

    return price

