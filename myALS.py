import pandas as pd
import numpy as np
from datetime import datetime

np.set_printoptions(threshold=np.nan, linewidth=100)
verbose = False

def getPi(i, R):

	P = R[:, i] > 0
	P[P == True] = 1
	P[P == False] = 0

	#transform True to 1, False to 0
	return P.astype(np.float64, copy=False)

def getPu(u, R):

	P = R[u] > 0
	P[P == True] = 1
	P[P == False] = 0

	#transform True to 1, False to 0
	return P.astype(np.float64, copy=False)

def getCi(i, α, R):
	return np.diag( 1 + α*R[:,i] )
	
def getCu(u, α, R):
	if( u < R.shape[1] ):
		return np.diag( 1 + α*R[u] )
	else:
		return np.array([])

#get ratings matrix
def getInput(factor, ratingsF, moviesF):
	#load data for table 3:
	ratings = pd.read_table(ratingsF, sep=',', dtype={'user_id': np.int32, 'movie_id': np.int32, 'rating': np.float64, 'timestamp': str})
	movies = pd.read_table(moviesF, sep=',', dtype={'movie_id': np.int32, 'title': str, 'title': str})

	#join all tables: tags, ratings and movies
	df = ratings.merge(movies, on=['movie_id'])

	if( verbose ):
		print('MOVIE LENS DATASET')
		print( df.head() )

	
	#moviesDict format: {user_id: {title:.., genres: ..}}
	moviesDict = {}
	userDict = {}
	for i in range(df.shape[0]):
		movie_id = df['movie_id'][i]
		moviesDict.setdefault(movie_id, {'title': df['title'][i], 'genre': df['genres'][i]})

		user_id = df['user_id'][i]
		userDict.setdefault(user_id, [])
		userDict[user_id].append( {'movie_id': movie_id, 'rating': df['rating'][i]} )
	

	#transpose table e.g: https://stackoverflow.com/q/28337117
	userItem = pd.pivot_table(df, columns=['movie_id'], index=['user_id'], values='rating')
	# Replace NaN with zeros
	userItem = userItem.fillna(0)

	
	data = {}
	data['R-observations'] = userItem.values
	data['m-users'], data['n-items'] = userItem.shape
	data['details'] = {'movies': moviesDict, 'users': userDict, 'ratings-movie': df}

	print( 'Dataset:', data['m-users'], 'users', 'by', data['n-items'], 'items' )

	'''
		data['X-user-factor-matrix'] = 5 * np.random.rand( data['m-users'],factor)
		data['Y-item-factor-matrix'] = 5 * np.random.rand( data['n-items'], factor)

		print()
		print( data['m-users'], 'users' )
		print( data['n-items'], 'items' )
		print( 'factors:', factor )
		print()
		
		print( 'R (observations):' )
		print(data['R-observations'])
		print()

		print( 'user factor matrix:', data['X-user-factor-matrix'].shape )
		print( data['X-user-factor-matrix'] )
		print( 'Y (item factor matrix):', data['Y-item-factor-matrix'].shape )
		print( data['Y-item-factor-matrix'] )
		print()

		λ = 40
		u = 1
		i = 1
		Cu = getCu( u, λ, data['R-observations'] )
		print('Cu@' + str(u))
		print( Cu )
		
		Ci = getCi( i, λ, data['R-observations'] )
		print('Ci@' + str(i))
		print(Ci)
		print()

		print('I')
		I = np.eye(data['n-items'])
		print(I)
		print()

		print('P(u='+str(u)+'):')
		print( getPu(u, data['R-observations']) )
		print('P(i='+str(i)+'):')
		print( getPi(i, data['R-observations']) )
		print()
	'''

	return data


def ALS(X, Y, R, m, n, factor, α, λ, maxIter):

	if( verbose ):
		print('\nR (USER ITEM TABLE)')
		print(R)
		print()

		print('m:', m, 'users')
		print('n:', n, 'items')
		print()
		print('X (user factor matrix)')
		print(X)
		print('\nY (item factor matrix)')
		print(Y)
		print()

	for iteri in range(maxIter):

		#if( iteri % 10 == 0 ):
		print('iter:', iteri)

		xTx = X.T.dot(X)
		yTy = Y.T.dot(Y)
		
		λI = np.eye(factor) * λ

		#X.shape[0] = m
		for u in range(m):
			
			Cu = getCu( u, α, R )
			CuI = Cu - np.eye(n)

			yT_CuI_y = Y.T.dot(CuI).dot(Y)
			pu = getPu(u, R)
			
			A = yTy + yT_CuI_y + λI
			b = Y.T.dot(Cu).dot(pu)
			xu = np.linalg.solve(A, b)			
			X[u] = xu

		for i in range(n):
			Ci = getCi( i, α, R )
			CiI = Ci - np.eye(m)
			xT_CiI_x = X.T.dot(CiI).dot(X)
			pi = getPi(i, R)

			A = xTx + xT_CiI_x + λI
			b = X.T.dot(Ci).dot(pi)

			yi = np.linalg.solve(A, b)
			Y[i] = yi

	if( verbose ):
		print('estimated X:')
		print(X)
		print('estimated Y:')
		print(Y)

	return X, Y

def getDetailsForUser(table, user_id):
	
	for i in range(table.shape[0]):
		print(table['user_id'][i], )

def printRecommendations(recommendations, detailsTable, K=10):
	
	print()
	#print('recommendations.shape:', recommendations.shape)
	#print('movies:', len(detailsTable['movies']))

	sortedMovieIDs = sorted(detailsTable['movies'].keys())
	sortedUserIDs = sorted(detailsTable['users'].keys())
	for user in range(len(recommendations)):
		#user watched these movies
		user_id = sortedUserIDs[user]
		print('for user_id:', user_id)
		print('watched:')
		
		for movie in detailsTable['users'][user_id]:
			movie_id = movie['movie_id']
			print( '\t', detailsTable['movies'][movie_id]['title'] + '(' + detailsTable['movies'][movie_id]['genre'] + ')' )

		print('recommend K:', K)
		#user was recommended these movies
		for movie in range( len(recommendations[user]) ):
			rui = recommendations[user][movie]
			movie_id = sortedMovieIDs[movie]
			detailsTable['movies'][movie_id]['rui'] = round(rui, 4)

		sortedKeys = sorted( detailsTable['movies'], key=lambda movie_id:detailsTable['movies'][movie_id]['rui'], reverse=True )
		sortedKeys = sortedKeys[:K]

		for i in range(len(sortedKeys)):
			movie_id = sortedKeys[i]
			print( '\t', i+1, detailsTable['movies'][movie_id]['title'] + ' (' + detailsTable['movies'][movie_id]['genre'] + ')', detailsTable['movies'][movie_id]['rui'] ) 

		if( user > 9 ):
			break 

		print()

'''
	#user_id,movie_id,rating,timestamp
	userMovieDict = {}
	movieDedupSet = set()
	with open('data/ratings.ori.csv', 'r') as infile:
		infile.readline()
		for line in infile:

			oriLine = line.strip()
			line = line.strip().split(',')
			
			if( line[1] not in movieDedupSet ):
				userMovieDict.setdefault( line[0], {'movie': line[1], 'movie-detail': oriLine} )
				movieDedupSet.add(line[1])


	outfile = open('ratings.1000.csv', 'w')
	dedupSet = set()
	for user, movieDct in userMovieDict.items():
		
		if( user not in dedupSet ):
			dedupSet.add(user)
			outfile.write( movieDct['movie-detail'] + '\n' )

		if( len(dedupSet) == 1000 ):
			break
	outfile.close()
'''

prevNow = datetime.now()

λ = 150
α = 40
factor = 100
maxIter = 10

print('λ =', λ)
print('α =', α)
print('factor =', factor)
print('maxIter =', maxIter)
print()

ratingsF = 'data/ratings.1000.csv'
moviesF = 'data/movies.ori.csv'

data = getInput(factor, ratingsF, moviesF)

X = 5 * np.random.rand(data['m-users'], factor)
Y = 5 * np.random.rand( data['n-items'], factor)

X, Y = ALS(X, Y, data['R-observations'], data['m-users'], data['n-items'], factor, α, λ, maxIter)

recommendations = X.dot(Y.T)
printRecommendations(recommendations, data['details'])

delta = datetime.now() - prevNow
print('\tdelta seconds:', delta.seconds)