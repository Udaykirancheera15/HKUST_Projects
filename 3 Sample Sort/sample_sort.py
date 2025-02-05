from pyspark import SparkContext
from bisect import bisect_right
import random

# Initialize SparkContext
sc = SparkContext("local", "SampleSort")

# Step 1: Define variables
n = 100  # Size of RDD
p = 8    # Number of partitions
s = 4    # Sample size factor

# Step 2: Prepare the data
r = sc.parallelize([random.randrange(1, n * 100, 1) for _ in range(n)], p)

# Step 3: Take a sample of size s*p
sample = r.takeSample(False, s * p, 0)

# Step 4: Sort the sample
sample_sorted = sorted(sample)

# Step 5: Pick the splitters
splitters = [sample_sorted[i * s] for i in range(p)]
splitters = [0] + splitters  # Add 0 at the beginning for easier binary search

# Broadcast splitters to all workers
splitters_bc = sc.broadcast(splitters)

# Step 6: Split items into buckets
def split_into_buckets(iterator):
    """
    Assigns each item in a partition to the appropriate bucket based on splitters.
    """
    splitters = splitters_bc.value
    buckets = [[] for _ in range(p)]  # Initialize empty buckets
    for item in iterator:
        # Find the correct bucket using binary search
        bucket_id = bisect_right(splitters, item) - 1
        # Ensure bucket_id is within valid range
        bucket_id = min(bucket_id, p - 1)
        buckets[bucket_id].append(item)
    # Yield tuples of (bucket_id, list of items)
    for bucket_id, bucket_items in enumerate(buckets):
        yield (bucket_id, bucket_items)

# Apply the function to each partition
r_buckets = r.mapPartitions(split_into_buckets)

# Step 7: Shuffle and sort buckets
sorted_buckets = r_buckets.groupByKey().map(lambda x: (x[0], sorted(x[1])))

# Step 8: Concatenate sorted buckets into one RDD
sorted_rdd = sorted_buckets.flatMap(lambda x: x[1])

# Step 9: Collect and print the sorted RDD
result = sorted_rdd.collect()
print("Sorted RDD:", result)

# Stop SparkContext
sc.stop()
