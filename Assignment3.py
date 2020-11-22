# Imports
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import time
from numpy.linalg import norm

#%% Load & premodify all necessary data
#root = '/Users/mueed/Documents/GitHub/DATA.ML.360_Assignment3/ml-latest-small/'
root = '/home/tuomas/Python/DATA.ML.360/Assignment2/ml-latest-small/'

df_movies = pd.read_csv(root+'movies.csv', usecols=['movieId', 'title'],
                        dtype={'movieId':'int32', 'title':'str'})
df_ratings = pd.read_csv(root+'ratings.csv', usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId':'int32', 'movieId':'int32', 'rating':'float32'})

df_movie_features = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Create sparse feature matrix out of dataframe
mat_movie_ratings = csr_matrix(df_movie_features.values)
# Use this to get corresponding movie-id
all_movie_ids = df_movie_features.columns

#%% Calculate rating means for all users
rating_sums = np.asarray(mat_movie_ratings.sum(axis=1)).ravel()
counts = np.diff(mat_movie_ratings.indptr)
user_means = (rating_sums / counts).ravel()

#%% Calculate similarities between users (Pearson correlation)
def pearson_correlation(u1_vec, u1_mean, u2_vec, u2_mean):
    u1_meanvec = np.ones(u1_vec.shape[0]) * u1_mean
    u1_diffvec = u1_vec - u1_meanvec
    
    u2_meanvec = np.ones(u1_vec.shape[0]) * u2_mean
    u2_diffvec = u2_vec - u2_meanvec
    
    return (u1_diffvec@u2_diffvec) / (norm(u1_diffvec) * norm(u2_diffvec))

normalize = True
sim_matrix = np.zeros((609 + 1, 609 + 1))
cases_matrix = np.zeros((609 + 1, 609 + 1))
not_found = -1 if normalize else 0
start = time.time()
for uid1 in range(610):
    for uid2 in range(uid1, 610):
        # Comparing user to itself.
        if(uid1==uid2): 
            sim_matrix[uid1, uid2] = not_found
            cases_matrix[uid1, uid2] = 0
            continue
        
        u1_ratings = mat_movie_ratings.getrow(uid1).toarray().ravel()
        u2_ratings = mat_movie_ratings.getrow(uid2).toarray().ravel()
        common_ratings = np.nonzero(u1_ratings * u2_ratings)
        u1_ratings = u1_ratings[common_ratings]
        u2_ratings = u2_ratings[common_ratings]
        
        pc = pearson_correlation(u1_ratings, user_means[uid1], u2_ratings, user_means[uid2])
        if(np.isnan(pc)):
            pc = not_found
        
        sim_matrix[uid1, uid2] = pc
        cases_matrix[uid1, uid2] = u1_ratings.shape[0]
        
    print("User {} ready".format(uid1))
print("Run time = {} min".format(round((time.time() - start) / 60.0, 1)))

# Normailze sim matrix
if(normalize):
    sim_matrix += 1
    sim_matrix *= 0.5
    
# Copy upper triangle to lower triangle
ltr_idx = np.tril_indices(sim_matrix.shape[0], -1)
sim_matrix[ltr_idx] = sim_matrix.T[ltr_idx]
cases_matrix[ltr_idx] = cases_matrix.T[ltr_idx]

#%% Optional assumptions. Set sim_matrix value to 0 if similarity score
#   was calculated from too few samples.
filt = cases_matrix < 1 # This setting makes no effect
sim_matrix[filt] = 0

#%% Find similar users
def get_n_largest_idx(vec, n=1):
    idxs = (-vec.astype('float')).argsort()[:n]
    return idxs

user_id = 0
N = 10
sim_users = get_n_largest_idx(sim_matrix[user_id,:],N)
print("{} most similar users for user {}".format(N, user_id+1))
for n in range(N):
    print('User {}, similarity score: {}, cases: {}'.format(sim_users[n]+1, sim_matrix[user_id,sim_users[n]],
                                                            cases_matrix[user_id,sim_users[n]]))
    
#%% User based collaborative filtering
def predict_rating(uid, mid):
    ratings = mat_movie_ratings.getcol(mid).toarray().ravel()
    rated_user_ids = np.nonzero(ratings)
    rated_user_ratings = ratings[rated_user_ids]
    # Get the mean ratings of rated users
    rated_user_means = user_means[rated_user_ids]  
    # Calculate rating prediction
    mean_diff = rated_user_ratings - rated_user_means
    sim_vector = sim_matrix[uid, rated_user_ids]
    return user_means[uid] + (sim_vector@mean_diff) / sim_vector.sum()
    #return user_means[uid] + (sim_vector@rated_user_ratings) / np.abs(sim_vector).sum()

# Predict rating scores for all unrated movies for a single user.
def predict_unrated(uid):
    all_ratings = mat_movie_ratings.getrow(uid).toarray().ravel()
    unrated_movies = np.where(all_ratings == 0)[0]
    # Predict scores for all unseen movies
    pred_ratings = []
    for movie_idx in unrated_movies:
        pred = predict_rating(uid, movie_idx)
        pred_ratings.append(pred)
        
    return np.array(pred_ratings).ravel(), np.array(unrated_movies).astype('int')

user_id = 0
pred_ratings, movie_idxs = predict_unrated(user_id)

#%% Calculate rating predictions for group individuals
user_group = np.array([0,1,2])
group_pred_ratings = []
group_movie_idxs = []
for uid in user_group:
    pred_ratings, movie_idxs = predict_unrated(uid)
    group_pred_ratings.append(pred_ratings)
    group_movie_idxs.append(movie_idxs)
    print('User {} ready.'.format(uid))

group_pred_ratings = np.array(group_pred_ratings, dtype='object')
group_movie_idxs = np.array(group_movie_idxs, dtype='object')

#%% Consider only ratings of common unrated movies
# Part 1. Create a mask.
group_common = np.intersect1d(group_movie_idxs[0],group_movie_idxs[1])
for i in range(2,group_movie_idxs.shape[0]):
    group_common = np.intersect1d(group_common, group_movie_idxs[i])

# Use this as a mask
group_common_idxs = []
for gmi in group_movie_idxs:
    common_idx = np.in1d(gmi, group_common)
    group_common_idxs.append(common_idx)   
    
group_common_idxs = np.array(group_common_idxs, dtype='object')

# Part 2. Get corresponding ratings for common unrated movies.
group_common_ratings = []
for i in range(group_common_idxs.shape[0]):
    mask = group_common_idxs[i]
    user_common_ratings = group_pred_ratings[i][mask]
    group_common_ratings.append(user_common_ratings)

# Store all common ratings and corresponding movies
# Use these in average method and least misery method
group_common_ratings = np.array(group_common_ratings, dtype='float')
group_common_movie_idxs = group_movie_idxs[0][group_common_idxs[0]]

#%%
''' Assignment 3 starts '''
#%% Single user Satisfcation

# Create a weight vector based on group statisfaction scores
# Multiple methods can be used
def calc_weights1(sat_scores):
    weights = sat_scores**(-1)
    weights /= np.sum(weights)   
    return weights

def calc_weights2(sat_scores):
    weights = 1-sat_scores
    weights /= np.sum(weights)   
    return weights

def weighted_avgs(sat_scores, group_common_ratings):
    # Transform weight array to column vector
    weight_vec = calc_weights1(sat_scores)
    print('Weight vec : {}'.format(weight_vec))
    # Calculate group score for each item
    group_ratings = np.matmul(weight_vec, group_common_ratings)
    return group_ratings

def weighted_avgs2(sat_scores, group_common_ratings):
    return group_common_ratings.mean(axis=0)

def group_list_sat(user, Grj):
    user_ratings = Grj[user]      
    return user_ratings.sum()
        
def user_list_sat(user, Auj, N=3):
    user_ratings = Auj[user]
    top_filter = get_n_largest_idx(user_ratings, N)
    print('In user_list_sat, user {} ratings = {}'.format(user, user_ratings[top_filter]))
    return user_ratings[top_filter].sum()

# Store group recommendations for each iteration
group_recommendations_ratings_iter = []
group_recommendations_idxs_iter = []
group_individual_ratings_iter = []
# Store statisfaction scores for each iteration
group_statisfactions_iter = []
gcr_current = []
# Initialize user statisfaction scores to equal for the first iteration
sat_scores = np.ones(user_group.shape[0]) * user_group.shape[0]**(-1)
group_statisfactions_iter.append(sat_scores)
# Iterate over this list
group_common_ratings_current = group_common_ratings
group_common_movie_idxs_current = group_common_movie_idxs
group_pred_ratings_current = group_pred_ratings
iterations = 3
N = 3
#%%
for j in range(iterations):
    print('Iteration {}'.format(j))
    # 1. Get a list of recommendations for group.
    # 1.1 Get recommendations for group using some aggeration method
    sat_scores_prev = group_statisfactions_iter[len(group_statisfactions_iter)-1]
    group_ratings = weighted_avgs(sat_scores_prev, group_common_ratings_current)
    # 1.2 Get top items
    top_mask = get_n_largest_idx(group_ratings, N)
    top_common_ratings = group_ratings[top_mask]
    top_common_idxs = group_common_movie_idxs_current[top_mask]
    # 1.3 Get also individual user ratings for group top items
    group_indiv_ratings = group_common_ratings_current[:,top_mask]
    # 1.4 Save rating scores & suggested items in each iteration
    group_recommendations_ratings_iter.append(top_common_ratings)
    group_recommendations_idxs_iter.append(top_common_idxs)
    group_individual_ratings_iter.append(group_indiv_ratings)
    
    # 2. Calculate how satisfied each group member is with suggestions in current iteration
    # 2.1 Calculate satisfaction scores for each user.
    satisfactions = []
    for i in range(user_group.shape[0]):
        print('------------------------')
        gls = group_list_sat(i, group_common_ratings_current[:,top_mask])
        uls = user_list_sat(i, group_common_ratings_current)
        print('gls / uls = {} / {}'.format(gls, uls))
        satisfactions.append(gls / uls)
        
    print('------------------------')
    print('satisfactions = {}'.format(satisfactions))  
    group_statisfactions_iter.append(np.array(satisfactions))
    
    # 3. Update group individual rating list by removing the ones already processed.
    # Do this also for item index list
    print('group_common_ratings_current size {}'.format(group_common_ratings_current.shape[1]))
    print('group_common_movie_idxs size {}'.format(group_common_movie_idxs_current.shape[0]))
    gcr_current.append(group_common_ratings_current)
    group_common_ratings_current = np.delete(group_common_ratings_current, top_mask, axis=1)
    group_common_movie_idxs_current = np.delete(group_common_movie_idxs_current, top_mask, axis=0)
    print()
    
#%% Print suggested items for a group in each iteration

# Print the results
print('Sequential group recommendations using weighted group average.')
print('List of {} most relevant movies in {} iterations for user group {}:'.format(N,iterations,user_group+1))
print('------------------------------------------------------')
for i in range(iterations):
    print('Iteration {}'.format(i+1))
    group_scores = group_recommendations_ratings_iter[i]
    group_suggestions_idx = group_recommendations_idxs_iter[i]
    for g_score, g_sugg_idx in zip(group_scores, group_suggestions_idx):
        movie_id = all_movie_ids[g_sugg_idx]
        movie_name = df_movies.loc[df_movies['movieId']==movie_id].get('title').values[0]
        print('Pred. rating: {}'.format(g_score))
        print('(Id : {}), {}\n'.format(movie_id, movie_name))
    
    print()
        
