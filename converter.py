import pm4py

# Read the .xes.gz log
log = pm4py.read_xes("BPI_Challenge_2012.xes.gz")

# Convert to pandas dataframe
df = pm4py.convert_to_dataframe(log)

# Save to CSV
df.to_csv("./BPIC_12.csv", index=False)
