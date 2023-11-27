
#####################################

import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

final_transaction_df = pd.read_csv('transaction.csv')
product_df = pd.read_csv('product.csv')

##################test distribution#######################
transaction_df_seg = final_transaction_df.merge(products_df, on='product_id', how='left')
transaction_df_seg = transaction_df_seg.sort_values(['transaction_date'])


prduct_seg = ["Wellness Essentials", "Chic & Trendy",
      "Home Comforts", "Gourmet Delights",
      "Family Essentials", "Tech & Gadgets"]

def segment_customers(transaction_df, product_df,
                      transaction_times_thresh, 
                      avg_qty_thresh, prduct_seg):
    
    # Step 1: Merge transaction_df and product_df and sort by transaction_date
    
    product_df_cp = product_df[product_df.segment.isin(prduct_seg)]
    df_seg = transaction_df.merge(product_df_cp, on='product_id', how='inner')
    df_seg = df_seg.sort_values(['transaction_date'])

    # Step 2: Group by customer_id and segment, calculate transaction count, and pivot the result
    df_seg_sum = df_seg.groupby(['customer_id', 'segment']).size().reset_index(name='transaction_count')
    df_seg_tran_num = df_seg_sum.pivot(index='customer_id', columns='segment', values='transaction_count').fillna(0)
    df_seg_tran_num = df_seg_tran_num.reset_index()

    # Step 3: Group by customer_id and segment, calculate sum of transaction quantity, and pivot the result
    df_seg_sum = df_seg.groupby(['customer_id', 'segment'])['quantity'].sum().reset_index(name='transaction_quantity')
    df_seg_quantity = df_seg_sum.pivot(index='customer_id', columns='segment', values='transaction_quantity').fillna(0)
    df_seg_quantity = df_seg_quantity.reset_index()

    # Step 4: Filter segments based on thresholds and create a list of customer segments
    cus_segment_lst = []
    for seg in set(product_df_cp['segment']):
        df_seg_seg1 = df_seg_tran_num[['customer_id', seg]]
        df_seg_seg1 = df_seg_seg1[df_seg_seg1[seg] >= transaction_times_thresh]

        df_seg_seg2 = df_seg_quantity[['customer_id', seg]]
        df_seg_seg2 = df_seg_seg2[df_seg_seg2[seg] >= avg_qty_thresh]

        df_seg_seg1 = df_seg_seg1[['customer_id']]
        df_seg_seg2 = df_seg_seg2[['customer_id']]
        cus_segment = df_seg_seg1.merge(df_seg_seg2, on='customer_id', how='inner')
        cus_segment['segment_cus'] = seg
        cus_segment_lst.append(cus_segment)

    # Step 5: Concatenate and stack the customer segments
    final_result = pd.concat(cus_segment_lst).sort_values(by='customer_id')
    final_result = final_result.drop_duplicates(['customer_id', 'segment_cus'])

    return final_result

# Call the function with your transaction_df, product_df, transaction_times_thresh, and avg_qty_thresh

transaction_times_thresh, avg_qty_thresh = 2, 2
customer_seg = segment_customers(transaction_df, product_df, 
                   transaction_times_thresh, avg_qty_thresh, prduct_seg)

########################

def segment_product(transaction_df, product_df, customer_seg):
    # Step 1: Get unique customer segments
    segs = set(customer_seg['segment_cus'])

    # Create a dictionary to store the results
    result_dict = {}

    for segment_cus in segs:
        # Get the customers in the current segment
        customer_cusset = customer_seg[customer_seg['segment_cus'] == segment_cus]
        customer_cusset = customer_cusset.drop_duplicates(subset='customer_id')

        # Merge transaction data with customer segment
        transaction_df_seg = transaction_df.merge(customer_cusset, on='customer_id', how='inner')
        transaction_df_seg = transaction_df_seg.sort_values(['transaction_date'])

        # Step 1a: Calculate average quantity per product_id
        df_avg_quantity_seg = transaction_df_seg.groupby('product_id')['quantity'].mean().reset_index()
        df_avg_quantity_seg.rename(columns={'quantity': 'avg_quantity_seg'}, inplace=True)

        # Step 1b: Add 'avg_quantity' column using the total average
        df_avg_quantity_seg['avg_quantity'] = transaction_df.groupby('product_id')['quantity'].mean().reset_index()['quantity']

        # Step 1c: Add 'buyer_num' and 'nonbuyers_num' columns
        buyer_num = transaction_df_seg.groupby('product_id')['customer_id'].nunique().reset_index()
        buyer_num.rename(columns={'customer_id': 'buyer_num'}, inplace=True)
        df_avg_quantity_seg = df_avg_quantity_seg.merge(buyer_num, on='product_id', how='left')
                
        df_avg_quantity_seg['nonbuyers_num'] = df_avg_quantity_seg.apply(\
        lambda row: customer_cusset[~customer_cusset\
        ['customer_id'].isin(transaction_df_seg[transaction_df_seg['product_id'] == row['product_id']]\
        ['customer_id'])]['customer_id'].nunique(),   axis=1)

        # Step 1d: Add 'lift_qty' column
        df_avg_quantity_seg['lift_qty'] = df_avg_quantity_seg['avg_quantity_seg'] / df_avg_quantity_seg['avg_quantity']

        # Step 1e: Add 'lift_num' column
        df_avg_quantity_seg['buyer_ratio'] = df_avg_quantity_seg['buyer_num'] / df_avg_quantity_seg['nonbuyers_num']

        # Step 1f: Filter records where lift_qty > 1
        df_avg_quantity_seg = df_avg_quantity_seg[df_avg_quantity_seg['lift_qty'] > 1]

        # Step 1g: Sort by lift_qty in descending order
        df_avg_quantity_seg = df_avg_quantity_seg.sort_values(by='lift_qty', ascending=False)

        # Store the result in the dictionary
        result_dict[segment_cus] = df_avg_quantity_seg

    return result_dict

# Call the function with your transaction_df, product_df, and customer_seg
resulting_dataframes = segment_product(transaction_df, product_df, customer_seg)

######################################################################

num_thresh_buy_ratio = 15
def update_product_segment(product_df, result_dict, num_thresh_buy_ratio):
    
    product_df_update = product_df.copy()
    segments = list(result_dict.keys())
    product_exists = list(set(product_df[product_df.segment.isin(segments)]['product_id']))
    segments_df_lst = []
    for itv in segments:
        seg_df = result_dict[itv]
        seg_df = seg_df[seg_df['product_id'].isin(product_exists) == False]
        seg_df = seg_df.sort_values(by='buyer_ratio', ascending=False)
        if len(seg_df) >= num_thresh_buy_ratio:
            seg_df = seg_df.head(num_thresh_buy_ratio)
            seg_df['segment'] = itv
            seg_df = seg_df[['product_id', 'lift_qty', 'buyer_ratio', 'segment']]
            segments_df_lst.append(seg_df)
        else:    
            seg_df = pd.DataFrame([])
    
    segments_df = pd.DataFrame([])        
    if len(segments_df_lst) > 0:
        segments_df = pd.concat(segments_df_lst)
        
        # Group by student_id and find the max grade and corresponding subject
        segments_df = segments_df.groupby('product_id').apply(lambda x: x.loc[x['lift_qty'].idxmax()]).reset_index(drop=True)

        # Rename columns as required
        segments_df = segments_df.rename(columns={'lift_qty': 'max_lift'})
        segments_df = segments_df[['product_id', 'segment']]
        segments_df = segments_df[(segments_df.product_id.isnull() == False) & (segments_df.segment.isnull() == False)]
        segments_df = segments_df.drop_duplicates(['product_id'])
        add_lst = list(set(segments_df.product_id))
        product_df_update = product_df_update[product_df_update.product_id.isin(add_lst) == False]
        product_df_update = pd.concat([product_df_update, segments_df])
        product_df_update = product_df_update.drop_duplicates(['product_id'])
        print ('The prodduct segments are updated')
    else:    
        print ('The prodduct segments are not updated')
    
    return product_df_update    

product_df_1 = update_product_segment(product_df, resulting_dataframes, num_thresh_buy_ratio)

c =  ["Wellness Essentials", "Chic & Trendy",
      "Home Comforts", "Gourmet Delights",
      "Family Essentials", "Tech & Gadgets"]

# set initial python list that contain the initial number
# of products with segment allocation   
segs =  ["Wellness Essentials", "Chic & Trendy",
      "Home Comforts", "Gourmet Delights",
      "Family Essentials", "Tech & Gadgets"]

product_df_update = product_df.copy()
product_assigned_n = len(set(product_df_update\
     [product_df_update.segment.isin(segs)]['product_id']))

product_assigned_lst = [product_assigned_n]
print ('Initially there are ' + str(product_assigned_n) +\
      ' products are assigned segment')
    
for j in range(10):
    #update product's segment 
    product_df_update = update_product_segment\
    (product_df_update, resulting_dataframes, num_thresh_buy_ratio)    
    #update customer segment       
    customer_seg_update = segment_customers\
    (transaction_df, product_df_update, transaction_times_thresh,\
    avg_qty_thresh, prduct_seg)
    #update product's segment for iteration: lift_qty and buyer_ratio 
    resulting_dataframes = segment_product\
    (transaction_df, product_df_update, customer_seg_update) 
    # get the number of products with allocated segment, add to list     
    product_assigned_n = len(set(product_df_update\
    [product_df_update.segment.isin(segs)]['product_id']))
    if product_assigned_n > max(product_assigned_lst):
        product_assigned_lst.append(product_assigned_n)
        print ('now there are ' +  str(product_assigned_n) + \
        ' products are assigned segment')
            

import matplotlib.pyplot as plt

# Number of products assigned to segments in each iteration
segment_allocation = [32, 54, 81, 108]

# Plotting the results
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(segment_allocation) + 1), segment_allocation)
plt.xlabel("Iteration")
plt.ylabel("Number of Products Assigned to Segments")
plt.title("Segment Allocation Progression with SegmentEx Algorithm")
plt.xticks(range(1, len(segment_allocation) + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotating the final value
final_value = segment_allocation[-1]
plt.annotate(f"Final Value: {final_value}", (len(segment_allocation), final_value), 
             textcoords="offset points", xytext=(-20, 10), ha='center', fontsize=12)

plt.show()

#####################################################################
def product_segment_prob(result_dict):
    result_dict = resulting_dataframes.copy()
    # Create an empty DataFrame with 'product_id'
    # result_dict = resulting_dataframes.copy()
    product_segment_df = pd.DataFrame({'product_id': result_dict[list(result_dict.keys())[0]]['product_id']})

    # Iterate through the result_dict values and merge them by 'product_id'
    df_lst = []
    
    for segment_cus, df in result_dict.items():
        # Select only the relevant columns and rename them
        selected_cols = ['product_id', 'lift_qty', 'buyer_ratio']
        renamed_cols = [col.replace('lift_qty', f'lift_qty_{segment_cus}').replace('buyer_ratio', f'lift_num_{segment_cus}') for col in selected_cols]
        df = df[selected_cols]
        df['segment'] = segment_cus
        df_lst.append(df)
        df.columns = renamed_cols + ['segment']

        # Merge the current DataFrame with the main product_segment_df
        product_segment_df = product_segment_df.merge(df, on='product_id', how='outer')
        product_segment_df = product_segment_df.fillna(0)
        
        
    df_all = pd.concat(df_lst)
    df_all.head(5)
    
    return product_segment_df

# Call the function with the resulting dictionary
product_segment_df = product_segment_prob(resulting_dataframes)




