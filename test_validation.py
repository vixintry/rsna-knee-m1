
import pandas as pd

from src.labels import TARGETS
from src.validation import validate_label_values

# Test 1: valid labels, including missing values
df = pd.DataFrame({target: [0, 1, None] for target in TARGETS})
validate_label_values(df)
print("PASS: Valid labels accepted")

# Test 2: invalid label must be rejected
df.loc[0, "ACL"] = 2

try:
    validate_label_values(df)
except ValueError as error:
    print("PASS: Invalid label rejected")
    print(error)
else:
    raise AssertionError("Invalid label was not rejected")
