import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

'''
ユーザ設定パラメータ
'''
# 学習方法変更用の定数
# 0:Q学習, 1:Sarsa, 2:Sarsa(λ)
FLAG = 0

# シミュレータ出力間隔
# n回に1回出力
SIMULATION_TERM = 50

'''
学習プログラム
'''
# パラメータ
MAP_W = 12
MAP_H = 6
# ε-greedy法のε
EPSILON = 0.10

# 学習率α
ALPHA = 0.10
# 割引率γ
GAMMA = 0.90
LAMBDA = 0.90
# ゴール時の報酬
GOAL_REWARD = 12
FALL_CLIFF_PENALTY = 100
# 1ステップ経過のペナルティ
ONE_STEP_PENALTY = 1
# 学習回数
LEANING_TIMES = 1000
# Qの初期値の最大値（乱数の最大値）
INIT_Q_MAX = 30

# 迷路
# 0 通れない 崖
# 1 通れる 道
# スタートは左下 MAZE[MAP_H - 1][1]
# ゴールは右下 MAZE[MAP_H - 1][MAP_W - 1]
# 配列は[y座標][x座標]なことに注意
MAZE = [
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]

# Q値の初期化
q = [[[0 for i3 in range(4)] for i2 in range(MAP_H)] for i1 in range(MAP_W)]
e = [[[0 for i3 in range(4)] for i2 in range(MAP_H)] for i1 in range(MAP_W)]

# エージェントの位置をスタート位置にセット
sPosX = 0
sPosY = MAP_H - 1
# エージェントが観測する位置をスタート位置にセット
sdPosX = 0
sdPosY = MAP_H - 1

# 方向 エージェントの行動
LEFT = 0;
UP = 1;
RIGHT = 2;
DOWN = 3;

sameA = [0] * 4

route = "Route\n"

def main():
	global sPosX, sPosY, sdPosX, sdPosY, route

	minStepNum = 2147483647

	count = 0
	for i in range(LEANING_TIMES):
		# エージェントの初期化
		initAgent()
		isGoal = False
		isCliff = False

		mem_r = 0

		stepNum = 0
		# print(i)
		while not(isGoal):
			if isCliff:
				# 崖に落ちたとき初期位置に戻しフラグを戻す
				sPosX = 0
				sPosY = MAP_H - 1
				isCliff = False

			stepNum += 1
			# print(stepNum)

			# print("sPosX: ", end="")
			# print(sPosX)
			# print("sPosY: ", end="")
			# print(sPosY)
			a = eGreedy()
			# print(a)
			# 収束してるかの確認用
			if i in range(LEANING_TIMES - 50, LEANING_TIMES):
				a = greedy()

			# 行動を行い次の状態を観測し報酬を受けとる
			r = Action(a)
			# ステップ経過のペナルティ
			r -= ONE_STEP_PENALTY

			if FLAG == 0:
				updateqQ(r, a)
			elif FLAG == 1:
				updatesQ(r, a)
			else:
				updatetQ(r, a)

			updateS()
			# plt.close()
			# mapGraphic(sPosX, sPosY)
			if i % SIMULATION_TERM == 0:
				plt.close()
				mapGraphic(sPosX, sPosY)

			if i == LEANING_TIMES - 1:
				plt.close()
				mapGraphic(sPosX, sPosY)


			# ゴールの判定
			if sPosX == MAP_W - 1 and sPosY == MAP_H - 1:
				isGoal = True

			if checkCliff():
				isCliff = True

			mem_r += r
			count += 1

		if stepNum < minStepNum:
			minStepNum = stepNum
			remRoute = route

		route = "Route\n"

	print(remRoute)
	printQ()
	print("Finished")
	print(count)



# 現在の位置での行動の選択 ε-greedy法
def eGreedy():
	global sPosX, sPosY, sdPosX, sdPosY
	selectedA = 0

	randNum = randint(100+1)

	if randNum >= EPSILON * 100.0:
		sameA = [0] * 4

		if checkSurround():
			count = 0

			for i in range(4):
				if sameA[i] == 1:
					count += 1

			randNum2 = randint(count)

			while randNum2 > 0:
				if sameA[selectedA] == 1:
					randNum2 -= 1
				selectedA += 1

		else:
			# εの確率 Q値が最大となるようなaを選択
			for a in range(4):
				if q[sPosX][sPosY][selectedA] < q[sPosX][sPosY][a]:
					selectedA = a

	else:
		# (1-ε)の確率 ランダムにaを選択
		selectedA = randint(4)

	return selectedA

def eGreedySd():
	global sPosX, sPosY, sdPosX, sdPosY
	selectedA = 0

	randNum = randint(100+1)

	if randNum <= EPSILON * 100.0:
		sameA = [0] * 4

		if checkSurroundSd():
			count = 0

			for i in range(4):
				if sameA[i] == 1:
					count += 1

			randNum2 = randint(count)

			while randNum2 > 0:
				if sameA[selectedA] == 1:
					randNum2 -= 1
				selectedA += 1
		else:
			# εの確率 Q値の最大となるようなaを選択
			for a in range(4):
				if q[sdPosX][sdPosY][selectedA] < q[sdPosX][sdPosY][a]:
					selectedA = a
	else:
		selectedA = randint(4)

	return selectedA

def greedy():
	global sPosX, sPosY, sdPosX, sdPosY
	selectedA = 0

	for a in range(4):
		if q[sPosX][sPosY][selectedA] < q[sPosX][sPosY][a]:
			selectedA = a

	return selectedA

def checkSurround():
	global sPosX, sPosY, sdPosX, sdPosY
	selectedA = 0

	for a in range(4):
		if q[sPosX][sPosY][selectedA] < q[sPosX][sPosY][a]:
			selectedA = a

	for i in range(4):
		if q[sPosX][sPosY][selectedA] == q[sPosX][sPosY][i]:
			sameA[i] = 1

	for i in range(4):
		if q[sPosX][sPosY][i] == 1:
			return True

	return False

def checkSurroundSd():
	global sPosX, sPosY, sdPosX, sdPosY
	selectedA = 0

	for a in range(4):
		if q[sdPosX][sdPosY][selectedA] < q[sdPosX][sdPosY][a]:
			selectedA = a

	for i in range(4):
		if q[sdPosX][sdPosY][selectedA] == q[sdPosX][sdPosY][i]:
			sameA[i] = 1

	for i in range(4):
		if q[sdPosX][sdPosY][i] == 1:
			return True

	return False

# directionの向きに移動可能であれば移動
# 報酬を返す
def Action(direction):
	global sPosX, sPosY, sdPosX, sdPosY, route

	r = 0

	# 観測後の状態に現在の状態を設定
	# この関数内で観測後のものに書き換える
	sdPosX = sPosX
	sdPosY = sPosY

	if direction == LEFT:
		# 移動可能か
		if sPosX > 0:
			sdPosX -= 1
			if MAZE[sPosY][sPosX - 1] == 0:
				# 崖に落ちたとき
				r -= FALL_CLIFF_PENALTY
		route += ("←" + "[" + str(sdPosX) + "][" + str(sdPosY) + "] \n")
	elif direction == UP:
		# 移動可能か
		if sPosY > 0:
			sdPosY -= 1
			if MAZE[sPosY - 1][sPosX] == 0:
				# 崖に落ちたとき
				r -= FALL_CLIFF_PENALTY
		route += ("↑" + "[" + str(sdPosX) + "][" + str(sdPosY) + "] \n")
	elif direction == RIGHT:
		# 移動可能か
		if sPosX < MAP_W - 1:
			sdPosX += 1
			if MAZE[sPosY][sPosX + 1] == 0:
				# 崖に落ちたとき
				r -= FALL_CLIFF_PENALTY
		route += ("→" + "[" + str(sdPosX) + "][" + str(sdPosY) + "] \n")
	elif direction == DOWN:
		# 移動可能か
		if sPosY < MAP_H - 1:
			sdPosY += 1
			if MAZE[sPosY + 1][sPosX] == 0:
				# 崖に落ちたとき
				r -= FALL_CLIFF_PENALTY
		route += ("↓" + "[" + str(sdPosX) + "][" + str(sdPosY) + "] \n")

	# ゴール報酬の設定
	if sdPosX == MAP_W - 1 and sdPosY == MAP_H - 1:
		r += (GOAL_REWARD + 1)

	return r

# Q値の更新 Q学習
def updateqQ(r, a):
	global sPosX, sPosY, sdPosX, sdPosY, q
	# 状態s'で行ったときにQ値が最大となるような行動
	maxA = 0
	for i in range(4):
		if q[sdPosX][sdPosY][maxA] < q[sdPosX][sdPosY][i]:
			maxA = i

	q[sPosX][sPosY][a] = q[sPosX][sPosY][a] + ALPHA * (r + GAMMA * q[sdPosX][sdPosY][maxA] - q[sPosX][sPosY][a])

def updatesQ(r, a):
	global sPosX, sPosY, sdPosX, sdPosY, q
	nextA = eGreedySd()

	q[sPosX][sPosY][a] = q[sPosX][sPosY][a] + ALPHA * (r + GAMMA * q[sdPosX][sdPosY][nextA] - q[sPosX][sPosY][a])


def updatetQ(r, a):
	global sPosX, sPosY, sdPosX, sdPosY, q, e
	nextA = eGreedySd()

	DELTA = r + GAMMA * q[sdPosX][sdPosY][nextA] - q[sPosX][sPosY][a];
	e[sPosX][sPosY][a] += 1;

	updateE(DELTA);

def updateS():
	global sPosX, sPosY, sdPosX, sdPosY
	sPosX = sdPosX
	sPosY = sdPosY

def printQ():
	for x in range(MAP_W):
		for y in range(MAP_H):
			for a in range(4):
				print("x:" + str(x) + " y:" + str(y) + " a:" + str(a) + " Q:" + str(q[x][y][a]))

def checkCliff():
	global sPosX, sPosY, sdPosX, sdPosY
	if sPosX >= 1 and sPosX <= MAP_W - 2:
		if sPosY == MAP_H - 1:
			return True
	return False

def initAgent():
	global sPosX, sPosY
	sPosX = 0
	sPosY = MAP_H - 1

def updateE(delta):
	global q, e
	for x in range(MAP_W):
		for y in range(MAP_H):
			for a in range(4):
				q[x][y][a] += ALPHA * delta * e[x][y][a];
				e[x][y][a] = GAMMA * LAMBDA * e[x][y][a];

def mapGraphic(x, y):
	fig = plt.figure()
	ax = plt.axes()

	start = patches.Rectangle(xy=(0, 5), width=1.0, height=1.0, fc="g")
	goal = patches.Rectangle(xy=(11, 5), width=1.0, height=1.0, fc="g")
	agent = patches.Rectangle(xy=(x, y), width=1.0, height=1.0, fc="b")
	ax.add_patch(start)
	ax.add_patch(goal)
	ax.add_patch(agent)

	hole01 = patches.Rectangle(xy=(1, 5), width=1.0, height=1.0, fc="#A9A9A9")
	hole02 = patches.Rectangle(xy=(2, 5), width=1.0, height=1.0, fc="#A9A9A9")
	hole03 = patches.Rectangle(xy=(3, 5), width=1.0, height=1.0, fc="#A9A9A9")
	hole04 = patches.Rectangle(xy=(4, 5), width=1.0, height=1.0, fc="#A9A9A9")
	hole05 = patches.Rectangle(xy=(5, 5), width=1.0, height=1.0, fc="#A9A9A9")
	hole06 = patches.Rectangle(xy=(6, 5), width=1.0, height=1.0, fc="#A9A9A9")
	hole07 = patches.Rectangle(xy=(7, 5), width=1.0, height=1.0, fc="#A9A9A9")
	hole08 = patches.Rectangle(xy=(8, 5), width=1.0, height=1.0, fc="#A9A9A9")
	hole09 = patches.Rectangle(xy=(9, 5), width=1.0, height=1.0, fc="#A9A9A9")
	hole10 = patches.Rectangle(xy=(10, 5), width=1.0, height=1.0, fc="#A9A9A9")
	ax.add_patch(hole01)
	ax.add_patch(hole02)
	ax.add_patch(hole03)
	ax.add_patch(hole04)
	ax.add_patch(hole05)
	ax.add_patch(hole06)
	ax.add_patch(hole07)
	ax.add_patch(hole08)
	ax.add_patch(hole09)
	ax.add_patch(hole10)

	plt.grid(which='major', linestyle='-')
	plt.xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
	plt.yticks([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
	plt.pause(.01)

if __name__ == "__main__":
	main()
