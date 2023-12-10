import numpy as np


def patch_match(img_a, img_b, patch_size=7, max_offset=10):
    # img_a and img_b should be (height x width) numpy arrays

    height, width = img_a.shape[:2]
    # Initialize random offset field
    offsets = np.dstack((np.random.randint(-max_offset, max_offset, size=(height, width)),
                         np.random.randint(-max_offset, max_offset, size=(height, width))))

    for iter in range(number_of_iterations):
        for y in range(height):
            for x in range(width):
                # Propagate: compare the offset with neighboring pixels
                # Assume the best offset is the one we have
                best_offset = offsets[y, x]
                best_cost = calculate_cost(img_a, img_b, x, y, best_offset, patch_size)

                # Check the pixel on the left
                if x > 0:
                    candidate_offset = offsets[y, x - 1]
                    candidate_cost = calculate_cost(img_a, img_b, x, y, candidate_offset, patch_size)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_offset = candidate_offset

                # Check the pixel above
                if y > 0:
                    candidate_offset = offsets[y - 1, x]
                    candidate_cost = calculate_cost(img_a, img_b, x, y, candidate_offset, patch_size)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_offset = candidate_offset

                # Random search: search around the best offset found so far
                for scale in range(max_offset, 0, -max_offset // 2):
                    random_offset = best_offset + np.random.randint(-scale, scale + 1, size=(2,))
                    random_cost = calculate_cost(img_a, img_b, x, y, random_offset, patch_size)
                    if random_cost < best_cost:
                        best_cost = random_cost
                        best_offset = random_offset

                # Update the offset field
                offsets[y, x] = best_offset

    # After iterating, offsets contain the field of best matches
    # You can then create a depth map based on these offsets
    return offsets


def calculate_cost(img_a, img_b, x, y, offset, patch_size):
    # Define the range of pixels to sample in each patch
    half_patch = patch_size // 2
    start_x = max(x - half_patch + offset[0], 0)
    end_x = min(x + half_patch + offset[0], width)
    start_y = max(y - half_patch + offset[1], 0)
    end_y = min(y + half_patch + offset[1], height)

    # Extract the patches
    patch_a = img_a[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
    patch_b = img_b[start_y:end_y + 1, start_x:end_x + 1]

    # Calculate the sum of squared differences cost between the two patches
    cost = np.sum((patch_a - patch_b) ** 2)

    return cost

# Please note that this code is a conceptual illustration and is not optimized for performance.
# In practice, there are many optimizations and details that would need to be addressed for a robust implementation.
