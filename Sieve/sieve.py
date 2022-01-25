import numpy as np
import rrcf

# ========================================
# Set tree parameters
# ========================================
num_trees = 10 # 50
tree_size = 10 # 128
windowSize_k = 10 # 50
threshold_h = 0.3

# ========================================
# Set global parameters
# ========================================
isMaintenanceStage = False
# Create a list to store attention score of each point
avg_attentionScore = []
forest = []
X = []    # before path vector or after


# A new trace come ...
while True:

    # ========================================
    # Path vector encoder
    # ========================================
    # path vector encoder 不在这里做！！！！！！
    newNode = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])



    if isMaintenanceStage == False:
        X.append(newNode)
        if len(X) == tree_size:
            # ========================================
            # Path vector encoder
            # ========================================
            # Get X
            X = [
            [1, 1, 1, 1, 1, 1,-1, 1, 1, 1],
            [2,-1, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3,-1, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5,-1, 5, 5, 5, 5],
            [1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
            [2, 2,-1, 2, 2, 3, 2, 2, 3, 2],
            [3, 3, 6, 3, 6, 3, 6,-1, 3, 6],
            [5, 6, 5, 5,-1, 6, 5, 5, 6, 5],
            [5, 7,-1, 7, 5, 7, 5, 5,-1, 7]
            ]

            # ========================================
            # Construction stage
            # ========================================
            # Create a forest
            for _ in range(num_trees):
                tree = rrcf.RCTree(X)
                forest.append(tree)
            isMaintenanceStage = True
    elif isMaintenanceStage == True:
        # ========================================
        # Path vector encoder
        # ========================================


        # Sum attention score of all RRCTs
        scoreSum = 0
        for i, tree in enumerate(forest):
            # ========================================
            # Maintenance stage
            # ========================================
            # Remove the oldest leaf
            tree.forget_point(min(tree.leaves.keys()))
            # Insert new path vector
            # No new dimension
            if len(newNode) == tree.ndim:
                tree.insert_point(newNode, index=max(tree.leaves.keys())+1)
            # Has new dimension
            else:
                # Extend leaves
                for leaf_index, leaf_value in tree.leaves.items():
                    tree.leaves[leaf_index].x = np.append(tree.leaves[leaf_index].x, [-1]*(len(newNode)-tree.ndim))
                    tree.leaves[leaf_index].b = tree.leaves[leaf_index].x.reshape(1, -1)
                
                tree.ndim = len(newNode)

                # Extend internal nodes
                tree._get_bbox_top_down(tree.root)

                # Build a new root
                node = tree.root
                parent = node.u
                leaf = rrcf.Leaf(x=newNode, i=max(tree.leaves.keys())+1, d=0)
                branch = rrcf.Branch(q=tree.ndim-1, p=-0.5, l=node, r=leaf,
                                    n=(leaf.n + node.n))
                # Set parent of new leaf and old branch
                node.u = branch
                leaf.u = branch
                # Set parent of new branch
                branch.u = parent
                # If a new root was created, assign the attribute
                tree.root = branch
                # Increment depths below branch
                tree.map_leaves(branch, op=tree._increment_depth, inc=1)
                # Increment leaf count above branch
                tree._update_leaf_count_upwards(parent, inc=1)
                # Update bounding boxes
                tree._tighten_bbox_upwards(branch)
                # Add leaf to leaves dict
                tree.leaves[max(tree.leaves.keys())+1] = leaf
            
            # ========================================
            # Calculate attention score
            # ========================================
            scoreSum = scoreSum + max(tree.leaves[leaf_index].d for leaf_index in tree.leaves.keys()) / tree.leaves[max(tree.leaves.keys())].d

        # Final score of a path vector
        scoreAvg = scoreSum / len(forest)

        # ========================================
        # Biased sampler
        # ========================================
        # A sliding window containing k most recent scores and the current score
        if len(avg_attentionScore) == windowSize_k+1:
            avg_attentionScore.pop(0)
        avg_attentionScore.append(scoreAvg)
        # Calculates the variance vark of the past k scores and the variance vark+1 of the k + 1 scores
        if len(avg_attentionScore) == windowSize_k+1:
            Var_k = np.var(avg_attentionScore[:-1])
            Var_k1 = np.var(avg_attentionScore)
            # Difference degree exceeds a threshold h
            if Var_k1 - Var_k > threshold_h:
                p = 1 / (1 + np.exp(2*np.mean(avg_attentionScore)-avg_attentionScore[-1]))
            # Otherwise ...
            elif Var_k1 - Var_k <= threshold_h:
                p = avg_attentionScore[-1] / np.sum(avg_attentionScore)
            # If p is no less than the random number, Sieve samples the trace
            if p >= np.random.uniform(0, 1):
                # Sample the trace
                print("Sample the trace ...")
            # Otherwise
            else:
                # Drop the trace
                print("Drop the trace ...")
        else:
            # Sliding window is not full
            print("Sliding window is not full, missing {} attention score ...".format(windowSize_k+1-len(avg_attentionScore)))