from pycaret.classification import *

# Load data
data = get_data('diabetes')

# Initialize setup
clf1 = setup(data, target='Class variable', session_id=123)