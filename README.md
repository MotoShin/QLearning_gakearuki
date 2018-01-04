# QLearning_gakearuki

動作環境  
  OS: macOS Sierra  
  バージョン: 10.12.6  
  言語: Python 3.5.2, Anaconda 4.2.0 (x86_64)  
  使用ライブラリ: numpy, matplotlib  
  
タスク「崖歩き」の説明  
  マップ構成（0:スタート, 1:安全地帯, 2:崖, 3:ゴール）  
    0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3  
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1    
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  
    
  スタートからゴールまでの最短距離を見つけさせるタスクです。
  ただし、スタートからゴールまでの最短距離は崖となっているため最短距離でのゴールは崖ギリギリのところを進む必要があります。
  
出力の説明  
  ・まず崖歩きのシミュレーションが立ち上がり、実際にエージェントの学習の様子が確認できます。  
  ・シミュレータは左上がスタートで右上がゴールとなっています。  
  ・スタートとゴールの間の灰色の部分は崖、青色はエージェントとなっています。  
  ・シミュレータが終了すると最も行動数が少なかったルートと、Q値が全て出力されます。  
  
注意点  
  ・環境が異なるとシミュレーションがうまく行かない可能性があります。  
  
ユーザ設定パラメータについて  
  ・FLAG:学習アルゴリズムの変更(0:Q学習,1:Sarsa,2:Sarsa(λ))が可能(2:Sarsa(λ)は未導入)  
  ・SIMULATION_TERM:シミュレータ起動間隔の設定が可能(全てシミュレートすると時間がかかる)
