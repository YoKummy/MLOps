import pandas as pd
import os
import sys

# 1. Read the new model's CSV
df = pd.read_csv('runs/detect/train/results.csv')
df.columns = df.columns.str.strip()
new_map = round(float(df.iloc[-1]['metrics/mAP50(B)']), 4)

# 2. Read the factory's High Score Board
prod_file = 'C:/Shadow_Pipeline_Production/best_map.txt'
if os.path.exists(prod_file):
    with open(prod_file, 'r') as f:
        prod_map = float(f.read().strip())
else:
    prod_map = 0.0 # First deployment gets an automatic pass
print(f'New Model mAP: {new_map}')
print(f'Factory mAP:   {prod_map}')

# 3. The Logic Gate
if new_map > prod_map:
    print('SUCCESS: Smarter model detected. Saving new high score.')
    with open('deployment_status.txt', 'w') as f: f.write('DEPLOY_MODEL')
    with open('new_high_score.txt', 'w') as f: f.write(str(new_map))
elif new_map == prod_map:
    print('NOTICE: Metrics are identical. This is a Code-Only update.')
    with open('deployment_status.txt', 'w') as f: f.write('DEPLOY_CODE')
else:
    print('REJECTED: New model is worse. Halting pipeline.')
    sys.exit(1) # This throws the Red X and stops everything!