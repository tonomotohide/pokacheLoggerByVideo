# -*- coding: utf-8 -*-
"""
https://pystyle.info/opencv-find-contours/
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
https://nehori.com/nikki/2021/04/01/post-27888/
"""

# GoogleDriveをマウント
from google.colab import drive
drive.mount('/content/gdrive')

# GoogleDriveから画像をコピー
!cp -r /content/gdrive/Othercomputers/'マイ パソコン'/'GoogleDrive(xxxxxxxx)'/pokerchase /content/images       # GoogleDrive上から画像ファイルをコピー

!ls /content/images     # 画像を確認

# OpenCV関連
!pip install opencv-python                      # opencvを導入
from google.colab.patches import cv2_imshow     # cv2.imshowがcoalboratoryで使えないのでサポートパッチのインポート

#  必要なモジュールをインポート
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 裏向きカードを定義・確認
img_path = '/content/images/facedowncard.jpg'

facedown_cards_rgb = cv2.imread(img_path)
facedown_cards_gray = cv2.imread(img_path,0)

cv2_imshow(facedown_cards_gray)

# ディーラーボタンを定義・確認
img_path = '/content/images/dealerbutton.jpg'
#img_path = '/content/gdrive/Othercomputers/マイ パソコン/GoogleDrive(xxxxxxxx)/pokerchase/dealerbutton.jpg'

dealerbutton_rgb = cv2.imread(img_path)
dealerbutton_gray = cv2.imread(img_path,0)

cv2_imshow(dealerbutton_gray)

# 各カードを定義・確認
img_path = '/content/images/A.jpg'

A_rgb = cv2.imread(img_path)
A_gray = cv2.imread(img_path,0)
A_mono = cv2.threshold(A_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(A_gray)

img_path = '/content/images/2.jpg'

two_rgb = cv2.imread(img_path)
two_gray = cv2.imread(img_path,0)
two_mono = cv2.threshold(two_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(two_gray)

img_path = '/content/images/3.jpg'

three_rgb = cv2.imread(img_path)
three_gray = cv2.imread(img_path,0)
three_mono = cv2.threshold(three_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(three_gray)

img_path = '/content/images/4.jpg'

four_rgb = cv2.imread(img_path)
four_gray = cv2.imread(img_path,0)
four_mono = cv2.threshold(four_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(four_gray)

img_path = '/content/images/5.jpg'

five_rgb = cv2.imread(img_path)
five_gray = cv2.imread(img_path,0)
five_mono = cv2.threshold(five_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(five_gray)

img_path = '/content/images/6.jpg'

six_rgb = cv2.imread(img_path)
six_gray = cv2.imread(img_path,0)
six_mono = cv2.threshold(six_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(six_gray)

img_path = '/content/images/7.jpg'

seven_rgb = cv2.imread(img_path)
seven_gray = cv2.imread(img_path,0)
seven_mono = cv2.threshold(seven_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(seven_gray)
cv2_imshow(seven_mono)

img_path = '/content/images/8.jpg'

eight_rgb = cv2.imread(img_path)
eight_gray = cv2.imread(img_path,0)
eight_mono = cv2.threshold(eight_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(eight_gray)

img_path = '/content/images/9.jpg'

nine_rgb = cv2.imread(img_path)
nine_gray = cv2.imread(img_path,0)
nine_mono = cv2.threshold(nine_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(nine_gray)

img_path = '/content/images/T.jpg'

T_rgb = cv2.imread(img_path)
T_gray = cv2.imread(img_path,0)
T_mono = cv2.threshold(T_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(T_gray)

img_path = '/content/images/J.jpg'

J_rgb = cv2.imread(img_path)
J_gray = cv2.imread(img_path,0)
J_mono = cv2.threshold(J_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(J_gray)

img_path = '/content/images/Q.jpg'

Q_rgb = cv2.imread(img_path)
Q_gray = cv2.imread(img_path,0)
Q_mono = cv2.threshold(Q_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(Q_gray)

img_path = '/content/images/K.jpg'

K_rgb = cv2.imread(img_path)
K_gray = cv2.imread(img_path,0)
K_mono = cv2.threshold(K_gray, 100, 255, cv2.THRESH_BINARY)[1]

cv2_imshow(K_gray)

# PyOCR
!apt install tesseract-ocr libtesseract-dev tesseract-ocr-jpn
!pip install pyocr

# 必要なモジュールをインポート
from PIL import Image
import pyocr

# pyocrが利用可能か確認
tools = pyocr.get_available_tools()
tool = tools[0]
print("Will use tool ‘%s'" % (tool.get_name()))     # Will use tool ‘Tesseract (sh) と表示されればOK

def ocr_from_image(img,language,layout):
    '''
    img     : cv2.imreadしたイメージ
    langage : どの言語として読むか(eng/jpn)
    layout  : OCRのモード設定(3/6)
    '''
    #cv2_imshow(img)
    ocr_result = tool.image_to_string(cv2pil(img), lang=language, builder=pyocr.builders.TextBuilder(tesseract_layout=layout))
    #print("OCR:",ocr_result)
    return ocr_result

def cv2pil(image):
    '''
     [Pillow ↔ OpenCV 変換](https://qiita.com/derodero24/items/f22c22b22451609908ee)
     OpenCV型 -> PIL型
     '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

# 全体画像からトーナメント情報を得る
def get_tournament_state(sc_img):
    '''
    sc_img  : cv2.imreadした全体スクリーンショット(1920*1080)
    '''
    # 初期化
    SB,BB,ante,max_players,stage = -1,-1,-1,-1,-1

    # SB/BBの取得
    blind_img = sc_img[15:50,210:495]
    blind_txt = ocr_from_image(blind_img,"eng",6)
    blind_info = [str(e) for e in blind_txt.split()]

    if blind_info[0] == "SB/BB" and len(blind_info) == 2:
        SB,BB = map(int,blind_info[1].split("/"))

    # anteの取得
    ante_img = sc_img[70:105,210:495]
    ante_txt = ocr_from_image(ante_img,"eng",6)
    ante_info = [str(e) for e in ante_txt.split()]

    if len(ante_info) == 2:
        ante = int(ante_info[1])

    result_dict = {"SB":SB,"BB":BB,"ante":ante}

    return result_dict

# ex. print(get_tournament_state(img_rgb))
# {'SB': 100, 'BB': 200, 'ante': 50, 'max_players': 6, 'stage': 6}

# 全体画像からトーナメントの進行状況を得る
ante_at_lv = [0,50,70,100,140,200,280,410,630,1400,2200,3200,4900]
sb_at_lv = [0,100,140,200,280,390,550,820,1250,2850,4300,6500,9800]
bb_at_lv = [0,200,280,400,560,780,1100,1640,2500,5700,8600,13000,19600]

def get_tournament_state(sc_img):
    '''
    sc_img  : cv2.imreadした全体スクリーンショット(1920*1080)
    '''
    # 初期化
    SB,BB,ante,max_players,stage = -1,-1,-1,-1,-1

    # SB/BBの取得
    blind_img = sc_img[15:50,210:495]
    blind_txt = ocr_from_image(blind_img,"eng",6)
    blind_info = [str(e) for e in blind_txt.split()]

    if blind_info[0] == "SB/BB" and len(blind_info) == 2:
        SB,BB = map(int,blind_info[1].split("/"))

    # anteの取得
    ante_img = sc_img[70:105,210:495]
    ante_txt = ocr_from_image(ante_img,"eng",6)
    ante_info = [str(e) for e in ante_txt.split()]

    if len(ante_info) == 2:
        ante = int(ante_info[1])

    if ante in ante_at_lv:
        lv = ante_at_lv.index(ante)
    else:
        lv = -1

    result_dict = {"SB":SB,"BB":BB,"ante":ante,"lv":lv}

    return result_dict

# 全体画像からトーナメントの固定情報を得る
entree_fee_list = [0.0,0.0,10.0,13.0,15.0,15.0,20.0]
starting_chip_list =  [0,6000,10000,10000,15000,20000,30000]

def get_tournament_info(sc_img):
    '''
    sc_img  : cv2.imreadした全体スクリーンショット(1920*1080)
    '''

    # stage名の取得
    stage_img = sc_img[1010:1060,215:615]
    stage_txt = ocr_from_image(stage_img,"eng",6)
    stage_txt = stage_txt.replace('|',"I")    # 誤読訂正

    if "VI" in stage_txt:
        stage = 6
        max_players = 6
    elif "IV" in stage_txt:
        stage = 4
        max_players = 6
    elif "V" in stage_txt:
        stage = 5
        max_players = 6
    elif "III" in stage_txt:
        stage = 3
        max_players = 6
    elif "II" in stage_txt:
        stage = 2
        max_players = 6
    elif "I" in stage_txt:
        stage = 1
        max_players = 4
    else:
        stage = -1
        max_players = 6

    if stage != -1:
        entree_fee = entree_fee_list[stage]
    else:
        entree_fee = -1

    if stage != -1:
        starting_chip = starting_chip_list[stage]
    else:
        starting_chip = -1

    result_dict = {"max_players":max_players,"stage":stage,"entree_fee":entree_fee,"starting_chip":starting_chip}

    return result_dict

# 全体画像から特定プレイヤーの画像を得る
def get_player_img(sc_img,seat_no):
    '''
    sc_img  : cv2.imreadした全体スクリーンショット(1920*1080)
    seat_no     : シート番号(Heroを1とし、時計回りに増加)
    '''
    if seat_no == 1:
        player_img = sc_img[595:880,575:1360]
    elif seat_no == 2:
        player_img = sc_img[500:720,60:390]
    elif seat_no == 3:
        player_img = sc_img[115:335,180:510]
    elif seat_no == 4:
        player_img = sc_img[15:235,800:1130]
    elif seat_no == 5:
        player_img = sc_img[115:335,1425:1755]
    elif seat_no == 6:
        player_img = sc_img[500:720,1545:1875]

    return player_img

# 特定プレイヤーの画像からプレイヤー名を得る
def get_player_name(sc_img,seat_no):
    '''
    readed_img  : cv2.imread→特定プレイヤーのみを切り取ったイメージ
    seat_no     : シート番号(Heroを1とし、時計回りに増加)
    '''
    player_img = get_player_img(sc_img,seat_no)     # 全体から特定プレイヤー部分だけ切り取る

    # 特定プレイヤー部分だけ切り取ったものからさらに名前部分だけ切り取る
    if seat_no != 1:
        name_img = player_img[182:218,70:310]
    else:
        name_img = player_img[230:270,450:760]

    gray_src = cv2.cvtColor(name_img, cv2.COLOR_BGR2GRAY)  # 画像をグレースケールにする

    # 拡大処理
    ex_temp = 3
    gray_src = cv2.resize(gray_src, None, interpolation=cv2.INTER_LINEAR, fx = ex_temp, fy = ex_temp)

    # 二値変換。前処理を使用しなかった場合は、blur_srcではなくgray_srcに書き換えるする
    mono_src = cv2.threshold(gray_src, 50, 255, cv2.THRESH_BINARY_INV)[1]      # 2）THRESH_BINARY_INV （しきい値=225）

    player_name = ocr_from_image(mono_src,"jpn",6)

    return player_name

# 全体画像からプレイヤー名一覧を得る
def get_player_name_dict(sc_img):
    '''
    sc_img  : cv2.imreadした全体スクリーンショット(1920*1080)
    '''
    # 初期化
    player_name_dict = {i:"Player"+str(i) for i in range(1,7)}

    # 各シート毎にプレイヤー名を取得
    player_name_dict[1] = "Hero"
    #for i in range(1,7):
    #    player_name_dict[i] = get_player_name(sc_img,i)

    return player_name_dict

# ex.
# print(get_player_name_list(img_rgb))
# {1: "username1", 2: 'username2', 3: 'username3', 4: 'username4', 5: 'username5', 6: 'username6'}

# 全体画像から特定プレイヤーのスタックを得る
def get_player_stack(sc_img,seat_no,BB,max_stack):
    '''
    sc_img  : cv2.imreadした全体のフルカラー画像
    seat_no : シート番号(Heroを1とし、時計回りに増加)
    '''
    #cv2_imshow(sc_img)

    player_img = get_player_img(sc_img,seat_no)     # 全体から特定プレイヤー部分だけ切り取る

    # 特定プレイヤー部分だけ切り取ったものからさらにスタック部分だけ切り取る
    if seat_no != 1:
        name_img = player_img[155:185,80:310]
    else:
        name_img = player_img[185:230,460:760]

    gray_src = cv2.cvtColor(name_img, cv2.COLOR_BGR2GRAY)  # 画像をグレースケールにする

    #cv2_imshow(gray_src)

    # 拡大処理
    ex_temp = 3
    gray_src = cv2.resize(gray_src, None, interpolation=cv2.INTER_LINEAR, fx = ex_temp, fy = ex_temp)

    # 二値変換。前処理を使用しなかった場合は、blur_srcではなくgray_srcに書き換えるする
    mono_src = cv2.threshold(gray_src, 100, 255, cv2.THRESH_BINARY_INV)[1]      # 2）THRESH_BINARY_INV （しきい値=225）

    #cv2_imshow(mono_src)

    player_stack_txt = ocr_from_image(mono_src,"eng",6)     # OCR処理
    player_stack = text_to_chip(player_stack_txt)

    if player_stack > max_stack:
        # 理論上の最大スタックを超えていたら小数点抜けとみなす
        player_stack = player_stack//10

    return player_stack

def text_to_chip(ocr_txt):
    # OCRしたテキストをチップ量の表記にする
    ocr_txt = ocr_txt.replace('O',"0")    # オーとゼロの誤読訂正
    ocr_txt = ocr_txt.replace(',',".")    # カンマと小数点の誤読訂正

    available_char = {"0","1","2","3","4","5","6","7","8","9","B"}
    checked_txt = ""

    for i in range(len(ocr_txt)):
        if ocr_txt[i] in available_char:
            checked_txt += ocr_txt[i]

    if "BB" in checked_txt:
        # BB表記
        checked_txt = checked_txt.replace('B',"")
        chip_float = float(checked_txt)
        chip_int = int(BB * chip_float)
    else:
        # Chip量表記
        try:
            checked_txt = checked_txt.replace('B',"8")
            int(checked_txt)
        except ValueError:
             chip_int = -1
        else:
            chip_int = int(checked_txt)

    return chip_int

# 全体画像から特定プレイヤーのアクション・状態を得る
def get_player_action(img_rgb,seat_no,BB):
    '''
    sc_img      : cv2.imreadした全体のフルカラー画像(1920*1080)
    seat_no     : シート番号(Heroを1とし、時計回りに増加)
    '''

    # 初期化
    is_dealed,action,bet_size = None,None,None                                      # アクション,ベットサイズ,カード配布の有無

    player_img_rgb = get_player_img(img_rgb,seat_no)                                # 全体から特定プレイヤー部分だけ切り取る
    player_img_gray = cv2.cvtColor(player_img_rgb, cv2.COLOR_BGR2GRAY)              # 画像をグレースケールにする

    if seat_no != 1:
        # Hero以外時の処理
        is_dealed = is_template_matched(player_img_gray,facedown_cards_gray,0.6)    # カード配布の有無の取得
    else:
        is_dealed = True

    if is_dealed:
        if seat_no != 1:
            action_img = player_img_gray[0:42,10:140]                               # アクション部分を切り出し
        else:
           action_img = player_img_gray[0:40,20:135]                                # アクション部分を切り出し

        mono_src = cv2.threshold(action_img, 100, 255, cv2.THRESH_BINARY_INV)[1]    # 二値変換。前処理を使用しなかった場合は、blur_srcではなくgray_srcに書き換えるする

        # 拡大処理
        ex_temp = 3
        mono_src = cv2.resize(mono_src, None, interpolation=cv2.INTER_LINEAR, fx = ex_temp, fy = ex_temp)

        action_txt = ocr_from_image(mono_src,"jpn",6)    # OCR処理

        if "フォールド" in action_txt or "フォールト" in action_txt:
            action = "Fold"
        elif "コール" in action_txt:
            action = "Call"
        elif "チェック" in action_txt:
            action = "Check"
        elif "ベット" in action_txt:
            action = "Bet"
        elif "レイズ" in action_txt:
            action = "Raise"
        elif "オール" in action_txt:
            action = "Allin"
        else:
            action = None

        if action != "Fold" and action != "Check" and  action != None:
            if seat_no == 1:
                bet_size_img = img_rgb[665:705,940:1090]
            elif seat_no == 2:
                bet_size_img = img_rgb[510:550,435:585]
            elif seat_no == 3:
                bet_size_img = img_rgb[355:395,495:645]
            elif seat_no == 4:
                bet_size_img = img_rgb[270:310,875:1025]
            elif seat_no == 5:
                bet_size_img = img_rgb[355:395,1290:1440]
            elif seat_no == 6:
                bet_size_img = img_rgb[515:555,1355:1505]

            bet_size_img = cv2.cvtColor(bet_size_img, cv2.COLOR_BGR2GRAY)                   # 画像をグレースケールにする
            mono_src = cv2.threshold(bet_size_img, 100, 255, cv2.THRESH_BINARY_INV)[1]      # 二値変換。前処理を使用しなかった場合は、blur_srcではなくgray_srcに書き換えるする

            bet_size_txt = ocr_from_image(mono_src,"eng",6)     # OCR処理
            bet_size = text_to_chip(bet_size_txt)

        else:
            # Check/Call時はbet_size = 0として扱う
            bet_size = 0

    player_action = {"is_dealed":is_dealed,"action":action,"bet_size":bet_size}

    return player_action

# 全体画像から特定プレイヤーにWINNER表示が出ているかを得る
def is_winner(img_rgb,seat_no):
    '''
    sc_img      : cv2.imreadした全体のフルカラー画像(1920*1080)
    seat_no     : シート番号(Heroを1とし、時計回りに増加)
    '''

    # 初期化
    player_img_rgb = get_player_img(img_rgb,seat_no)                                # 全体から特定プレイヤー部分だけ切り取る
    player_img_gray = cv2.cvtColor(player_img_rgb, cv2.COLOR_BGR2GRAY)              # 画像をグレースケールにする

    if seat_no != 1:
        action_img = player_img_gray[0:42,10:140]                                   # アクション部分を切り出し
    else:
        action_img = player_img_gray[0:40,20:135]                                   # アクション部分を切り出し

    mono_src = cv2.threshold(action_img, 100, 255, cv2.THRESH_BINARY_INV)[1]        # 二値変換

    # 拡大処理
    ex_temp = 3
    mono_src = cv2.resize(mono_src, None, interpolation=cv2.INTER_LINEAR, fx = ex_temp, fy = ex_temp)

    action_txt = ocr_from_image(mono_src,"eng",6)    # OCR処理
    action_txt = action_txt.replace('|',"I")         # 誤読訂正

    if "WIN" in action_txt or "VIN" in action_txt:
        return True
    else:
        return False

# 全体画像からボタンポジションを得る
def get_button_position(sc_img):
    '''
    readed_img  : cv2.imread→特定プレイヤーのみを切り取ったイメージ
    '''
    w, h = dealerbutton_gray.shape[::-1]

    t = [0,620,540,256,230,250,545]
    l = [0,1100,375,505,1045,1325,1455]

    for i in range(1,7):
        extracted_rgb = sc_img[t[i]:t[i]+120,l[i]:l[i]+120]               # DealerButtonがありえる場所周辺を抽出
        extracted_gray = cv2.cvtColor(extracted_rgb, cv2.COLOR_BGR2GRAY)  # 画像をグレースケールにする

        if is_template_matched(extracted_gray,dealerbutton_gray,0.8):
            return i
    else:
        return -1

# ターゲット画像にテンプレート画像が含まれるかを判定する
def is_template_matched(target_gray_img,template_gray_img,threshold):
    '''
    (両方ともグレスケ済みの)ターゲット画像にテンプレート画像が含まれるかを判定する
    threshold : 判定の閾値。0.0～1.0。ex.0.8
    '''
    #cv2_imshow(target_gray_img)
    #cv2_imshow(template_gray_img)

    w, h = template_gray_img.shape[::-1]

    res = cv2.matchTemplate(target_gray_img,template_gray_img,cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)

    cnt = 0

    for pt in zip(*loc[::-1]):
        # cv2.rectangle(target_gray_img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cnt += 1

    #cv2_imshow(target_gray_img)

    if cnt > 0:
        return True
    else:
        return False

def get_max_template_matched(target_gray_img,template_gray_img):
    '''
    (両方ともグレスケ済みの)ターゲット画像とテンプレート画像の最大類似度を得る
    '''

    target_gray_img = target_gray_img.astype(np.uint8)
    template_gray_img = template_gray_img.astype(np.uint8)

    res = cv2.matchTemplate(target_gray_img,template_gray_img,cv2.TM_CCOEFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

    return  maxVal

# 全体画像から、ボードのカードを得る
def get_board_cards_info(sc_img_rgb,card_place):
    '''
    sc_img_rgb : 全体画像(フルカラー)
    card_place : ボード中のカードの位置(1がFlopの最初、5がRiver)
    '''
    l = [0,655,785,912,1040,1170]

    board_card_rgb = sc_img_rgb[395:535,l[card_place]:l[card_place]+95]    # ボードのカード1枚を抽出

    b,g,r = board_card_rgb[20,90,:]

    if b >= 240 and g >= 240 and r >= 240:
        # カードの余白になる箇所が白色 = カードが配られている → カードを判別する

        #cv2_imshow(board_card_rgb)

        # テンプレートマッチングのために拡大処理
        ex_temp = 1.2
        board_card_rgb = cv2.resize(board_card_rgb, None, interpolation=cv2.INTER_LINEAR, fx = ex_temp, fy = ex_temp)
        board_card_rgb = board_card_rgb[5:105,5:105]

        board_card_num = get_card_number(board_card_rgb)
        board_card_suit = get_card_suit(board_card_rgb)

        if board_card_num != "0" and board_card_suit != "0":
            return str(board_card_num) + board_card_suit

    return -1

# 全体画像から、持っているカードを得る
def get_player_cards_info(sc_img,seat_no):

    if seat_no == 1:
        width = 100
        height = 100
        left_card_rgb = sc_img[695:695+height,780:780+width]
        right_card_rgb = sc_img[720:720+height,895:895+width]

    else:
        t = ["*","*",507,122,25,122,507]
        ll = ["*","*",225,345,967,1589,1708]
        lr = ["*","*",308,428,1052,1673,1790]
        width = 100
        height = 100

        left_card_rgb = sc_img[t[seat_no]:t[seat_no]+height,ll[seat_no]:ll[seat_no]+width]
        right_card_rgb = sc_img[t[seat_no]:t[seat_no]+height,lr[seat_no]:lr[seat_no]+width]

    # カードを判別する
    left_card_num = get_card_number(left_card_rgb)
    left_card_suit = get_card_suit(left_card_rgb)
    right_card_num = get_card_number(right_card_rgb)
    right_card_suit = get_card_suit(right_card_rgb)

    return [left_card_num+left_card_suit,right_card_num+right_card_suit]

# カードの画像から数字を判別する
def get_card_number(card_img_rgb):

    card_image_gray = cv2.cvtColor(card_img_rgb, cv2.COLOR_BGR2GRAY)  # 画像をグレースケールにする
    card_image_mono = cv2.threshold(card_image_gray, 220, 255, cv2.THRESH_BINARY)[1]

    card_no = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"]
    card_file = [A_mono,two_mono,three_mono,four_mono,five_mono,six_mono,seven_mono,eight_mono,nine_mono,T_mono,J_mono,Q_mono,K_mono]
    match = [get_max_template_matched(card_image_mono,card_file[i]) for i in range(13)]

    #cv2_imshow(card_image_mono)
    #print(card_no[match.index(max(match))],max(match),match)

    max_index = match.index(max(match))

    if max(match) >= 0.3:
        return card_no[max_index]
    else:
        return "0"

def get_card_suit(card_rgb_img):
    '''
    カード画像からスートを色で判別する
    '''
    h,w,c = card_rgb_img.shape

    y = h//4

    for x in range(3,w):
        b,g,r = card_rgb_img[y,x,:]

        if b <= 20 and g <= 20 and r <= 20:
            return "s"
        elif 225 <= b <= 255 and 55 <= g <= 100 and 13 <= r <= 53:
            return "d"
        elif 20 <= b <= 80 and 0 <= g <= 50 and 180 <= r <= 230:
            return "h"
        elif 50 <= b <= 90 and 110 <= g <= 180 and 0 <= r <= 65:
            return "c"

    return "0"

def init_tournament(img_rgb):
    '''
    トーナメント開始時のみ必要な処理を実施
    '''
    tournament_id = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')     # tournament IDは開始日時とする
    game_id = tournament_id                                                 # game IDは開始日時を0として扱う

    tournament_info = get_tournament_info(img_rgb)
    player_name_dict = get_player_name_dict(img_rgb)

    return tournament_id,tournament_info,player_name_dict

def convert_bet_to_pot(player_betting,is_folded):
    '''
    BET総額からpotをメイン/サイドに分割する
    betting_list        : [BET額,そのbetをしたプレイヤーのシート番号]を格納したリスト
    non_folded_player   : フォールドしていないプレイヤー一覧
    '''

    bet_list = [[player_betting[seat_no],seat_no] for seat_no in range(1,7)]  # BET総額,プレイヤーのリストを作る
    bet_list.sort(reverse=True)

    divided_pot = list()    # [pot額,参加者]を格納していく。左側の方が参加者が多いpotである

    while len(bet_list) >= 1:

        if bet_list[-1][0] == 0:
            # BET総額が0のプレイヤーは排除
            bet_list.pop(-1)
        else:
            # BET総額が小さい方からポットを作っていく
            bet = bet_list[-1][0]
            part_pot_chip = 0
            part_pot_player = list()

            for i in range(len(bet_list)):
                bet_list[i][0] -= bet
                part_pot_chip += bet

                if is_folded[bet_list[i][1]] == False:
                    # 金額を出してフォールドしていなければ獲得権利がある
                    part_pot_player.append(bet_list[i][1])

            if len(divided_pot) != 0 and divided_pot[-1][1] == part_pot_player:
                # 獲得権利があるプレイヤーが同一ならポットをまとめる
                divided_pot[-1][0] += part_pot_chip
            else:
                # 異なる場合はサイドポットにする
                divided_pot.append([part_pot_chip,part_pot_player])

    return divided_pot

# ex.
# test = [[30,1],[60,2],[180,3],[60,4],[500,2]]     # SB:1,BB:2  3:Raise All in 4:Call All in 2:All in Call
# print(convert_bet_to_pot(test,[True,True,False,False,False]))
# [[380, [2]], [240, [2, 3]], [210, [2, 3, 4]]]

import itertools
import collections

poker_hand_rank = ['*', 'High Card', 'a pair of ', 'two pair, ', 'three of a kind', 'a straight, ', 'a flush, ', 'a full house, ', 'a four of a kind', 'a straight flush', 'a royal Flush']

def get_strongest_hand_rank(dealt_card,board_card):
    '''
    ボードと手札から作れる最良役を返す

    戻り値： (役の種類、数字の並び)
    (役の種類はハイカードを1～(ロイヤル含む)ストレートフラッシュを9とする https://upswingpoker.com/poker-rules/)
    (数の並びではAは14 or 1で表記する。他は数字のまま)
    '''
    # 表記を数字,スートの形に直す
    converted_available_cards = list()
    available_cards = dealt_card + board_card
    for card in available_cards:
        card_num = card[0]
        card_suit = card[1]

        char_int = ["T","J","Q","K","A"]
        if card_num in char_int:
            card_num = 10 + char_int.index(card_num)
        else:
            card_num = int(card_num)

        converted_available_cards.append((card_num,card_suit))

    best_combination = [1,[7,5,4,3,2]]

    # 7枚から5枚選ぶ組み合わせを全て試す
    for choiced_cards in itertools.combinations(converted_available_cards,5):
        combination = get_hand_ranking(choiced_cards)

        # 作れる役と今までの最良役と比較する
        if combination[0] > best_combination[0]:
            best_combination = combination
        elif combination[0] == best_combination[0]:
            for i in range(5):
                if combination[1][i] > best_combination[1][i]:
                    best_combination = combination
                    break

    return best_combination


def get_hand_ranking(five_cards):
    '''
    5枚のカードで作れる役を返す
    '''

    suits = {"s":0,"h":0,"d":0,"c":0,}
    numbers = list()

    # スートと数字に分割
    for i in range(5):
        suits[five_cards[i][1]] += 1
        numbers.append(five_cards[i][0])

    numbers.sort(reverse=True)

    if suits["s"] == 5 or suits["h"] == 5 or suits["d"] == 5 or suits["c"] == 5:
        # 単一スート時の処理
        if numbers[0] - numbers[4] == 4:
            return [9,numbers]                      # Straight Flash
        elif numbers[0] == 14 and numbers[1] - numbers[4] == 3:
            return [9,numbers[1:] + [1]]   # Straight Flash
        else:
            return [6,numbers]  # Flash
    else:
        # 非単一スート時の処理
        counted = collections.Counter(numbers)
        cards_counts = {i:list() for i in range(5)}     # key:枚数 , value:そのカードの組み合わせ ex. card_counts[2] = [14,14,5,5]

        for k,v in counted.items():
            for i in range(v):
                cards_counts[v].append(k)

        if len(cards_counts[4]) == 1:
            return [8,cards_counts[4] + cards_counts[1]]
        elif len(cards_counts[3]) == 3:
            if len(cards_counts[2]) == 2:
                return [7,cards_counts[3] + cards_counts[2]]
            else:
                return [4,cards_counts[3] + cards_counts[1]]
        elif len(cards_counts[2]) == 4:
            return [3,cards_counts[2] + cards_counts[1]]
        elif len(cards_counts[2]) == 2:
            return [2,cards_counts[2] + cards_counts[1]]
        else:
            if numbers[0] - numbers[4] == 4:
                return [5,numbers]                      # Straight
            elif numbers[0] == 14 and numbers[1] - numbers[4] == 3:
                return [5,numbers[1:]+[1]]              # Straight(wheel)
            else:
                return [1,cards_counts[1]]

def is_card_dealt_to_hero(img_rgb):
    '''
    全体画像からheroにカードが配られているかを判定する
    '''
    b,g,r = img_rgb[730,850,:]

    if b >= 200 and g >= 200 and r >= 200:
        return "Yes(Not fold)"
    elif 140 >= b >= 120 and 140 >= g >= 120 and 140 >= r >= 200:
        return "Yes(folded)"
    elif 50 >= b >= 30 and 70 >= g >= 50 and 14 >= r >= 0:
        return "No"
    else:
        return "Other"

def is_card_showed(img_rgb,seat_no):
    '''
    全体画像からseat_noがカードをSHOWしているかを判定する
    '''
    if seat_no == 1:
        width = 100
        height = 100
        left_card_rgb = img_rgb[695:695+height,780:780+width]
        right_card_rgb = img_rgb[720:720+height,895:895+width]
    else:
        t = ["*","*",507,122,25,122,507]
        ll = ["*","*",225,345,967,1589,1708]
        lr = ["*","*",308,428,1052,1673,1790]
        width = 100
        height = 100
        left_card_rgb = img_rgb[t[seat_no]:t[seat_no]+height,ll[seat_no]:ll[seat_no]+width]
        right_card_rgb = img_rgb[t[seat_no]:t[seat_no]+height,lr[seat_no]:lr[seat_no]+width]

    b,g,r = left_card_rgb[30,70,:]

    if b >= 200 and g >= 200 and r >= 200:
        return True
    else:
        return False

from collections import defaultdict

poker_hand = ["","","",]

def get_winning(game_pot,hand_rank_list):
    '''
    potをプレイヤーのハンドに応じて分ける
    入力：
        game_pot        : [POT額,そのPOTの権利があるプレイヤーのシート番号一覧] を格納したリスト
        hand_rank_list  : 各プレイヤーのハンド((成立役名,カードの並び)とシート番号(強→弱 順)
    戻り値：
        winning         : 各プレイヤーの獲得ポット合計、獲得ポット種別が入る
    '''

    # hand_rank_listを同一ハンドの場合に一緒のランクになるようにする
    ranking_dict = defaultdict(list)
    for hand,comb,seat_no in hand_rank_list:
        hand_comb = str(hand) + ":" +  ','.join(map(str,comb))
        ranking_dict[hand_comb].append(seat_no)

    listed_ranking = [(k,v) for k,v in ranking_dict.items()]
    listed_ranking.sort(reverse=True)

    winning = [[0,list()] for i in range(7)]    # 獲ったポットの額、種類(Mainが0、サイドが1,2,3...)

    for pot_no in range(len(game_pot)):
        pot,gainable_seat_no = game_pot[pot_no]
        gainable_seat_no = set(gainable_seat_no)

        for j in range(len(listed_ranking)):
            players = listed_ranking[j][1]
            winner = set(players) & gainable_seat_no

            if len(winner) != 0:
                for seat_no in winner:
                    winning[seat_no][0] += pot//(len(winner))
                    winning[seat_no][1].append(pot_no)
                break

    return winning

number_str = ["*","Ace","Deuce","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Jack","Queen","King","Ace"]
# poker_hand_rank = ['*', 'High Card', 'a pair of ', 'two pair, ', 'three of a kind', 'a straight, ', 'a flush, ', 'a full house, ', 'a four of a kind', 'a straight flush', 'a royal Flush']

def conv_hand_to_str(hand,cards_combination):

    hand = int(hand)
    hand_txt = poker_hand_rank[hand]

    if hand == 9 or hand == 5:
        # straight (flush)
        high = number_str[cards_combination[0]]
        low = number_str[cards_combination[4]]
        hand_txt += high + " to " + low
    elif hand == 8 or hand == 4:
        # a four/three of a kind
        high = number_str[cards_combination[0]]
        hand_txt += high + "s"
    elif hand == 7:
        # full house
        high = number_str[cards_combination[0]]
        low = number_str[cards_combination[4]]
        hand_txt += high + "s full of " + low + "s"
    elif hand == 6:
        # flush
        high = number_str[cards_combination[0]]
        hand_txt += high + " high"
    elif hand == 3:
        # two pair
        high = number_str[cards_combination[0]]
        low = number_str[cards_combination[2]]
        hand_txt += high + "s and " + low + "s"
    elif hand == 2:
        # pair
        high = number_str[cards_combination[0]]
        hand_txt += "of " + high + "s"
    elif hand == 1:
        high = number_str[cards_combination[0]]
        hand_txt += high

    return hand_txt

import cv2
from collections import deque
from datetime import datetime, timedelta, timezone
import bisect
import time

number_in_roman = ["*","I","II","III","IV","V","VI","VII","VII","IX","X"]

# path = "/content/images/pokache_allin.mp4"
path = '/content/gdrive/Othercomputers/マイ パソコン/GoogleDrive(xxxxxx)/pokerchase/pokache_full.mp4'
cap = cv2.VideoCapture(path)

skip_frames = 5
sleep_time = 0.001

output_txt = list()

# ここ以後動画が終わるまで繰り返し

# ●トナメ毎の処理
# トーナメントの開始を待機

# プレイヤーに手札が配られていない状態まで待機
while True :

    for i in range(skip_frames):
        ret,img_rgb = cap.read()   #フレーム情報取得

    time.sleep(sleep_time)

    if ret == False:
        #動画が終われば処理終了
        raise Exception("End of movie")
    else:
        if is_card_dealt_to_hero(img_rgb) == "No":
            break


# トーナメント情報の取得
tournament_id,tournament_info,player_name_dict = init_tournament(img_rgb)   # tournament_info : {"max_players":max_players,"stage":stage,"entree_fee":entree_fee,"starting_chip":starting_chip}
entree_fee = tournament_info["entree_fee"]
max_players = tournament_info["max_players"]
stage = tournament_info["stage"]
starting_chip = tournament_info["starting_chip"]


# プレイヤーに手札が配られるまで待機
while True :
    #print("Frame: "+ str(i))

    for i in range(skip_frames):
        ret,img_rgb = cap.read()   #フレーム情報取得

    time.sleep(sleep_time)

    if ret == False:
        #動画が終われば処理終了
        raise Exception("End of movie")
    else:
        if is_card_dealt_to_hero(img_rgb) == "Yes(Not fold)":
            # 他プレイヤーへのカード配布されるのを待つ分、フレームを捨てる
            for i in range(skip_frames * 2):
                ret,img_rgb = cap.read()   #フレーム情報取得
                if ret == False:
                    #動画が終われば処理終了
                    raise Exception("End of movie")
            break


# トーナメント開始。動画終了まで継続()
while True:
    # 各ハンドの初期化
    hand_id = int(time.time() * 1000000)                # UnixTime(ミリ秒)をhand idとする

    waiting_action_players = deque()                    # action待ちのプレイヤーの一覧
    action_finished_players = list()                    # action済のプレイヤーの一覧
    allin_players = list()                              # all in済のプレイヤーの一覧

    dealt_card = [["X?","X?"] for i in range(7)]        # 各プレイヤーに配られたカード
    player_stack = [0 for i in range(7)]                # 各プレイヤーのスタック
    is_folded = [False for i in range(7)]               # フォールドしているか。したなら、いつフォールドしたか
    is_allin = [False for i in range(7)]                # オールインしているか

    is_allin = [False for i in range(7)]                # オールインしているか

    acquired = [0 for i in range(7)]                    # このハンドでの獲得額

    # ●Preflop時
    tournament_state = get_tournament_state(img_rgb)    # トーナメントの進行状況の取得
    BTN_pos = get_button_position(img_rgb)              # ボタン位置の取得
    SB = tournament_state["SB"]
    BB = tournament_state["BB"]
    ante = tournament_state["ante"]
    level = tournament_state['lv']

    # ex. PokerChase Game #20220123123456: Tournament #20220123123456, $20.0$0.00 Hold'em No Limit - Level V (390/780) - 2022/01/23 12:34:56JST
    line = "PokerChase Game #" + str(hand_id) + ": Tournament #" + str(tournament_id) + ", $" + str(entree_fee)  + "+$0.00 Hold'em No Limit - Level " + number_in_roman[level] + " (" + str(SB) + "/" + str(BB) + ") - " + datetime.now(timezone.utc).strftime('%Y/%m/%d %H:%M:%S') + " GMT"
    print(line)
    output_txt.append(line)

    # ex. Table '123456789 3' 9-max Seat #8 is the button
    table_name = "RANK MATCH STAGE-" + number_in_roman[stage]
    line = "Table '" + table_name + "' "+ str(max_players) + "-max Seat #" + str(BTN_pos) + " is the button"
    print(line)
    output_txt.append(line)

    # action待ちプレイヤーのリストを作成・スタックを取得
    max_stack = starting_chip * max_players

    for seat_no in range(1,7):
        player_action = get_player_action(img_rgb,seat_no,BB)

        if player_action["is_dealed"]:
            waiting_action_players.append(seat_no)
            player_stack[seat_no] = get_player_stack(img_rgb,seat_no,BB,max_stack)
        else:
            is_folded[seat_no] = "Finished"
            player_stack[seat_no] = 0

    print()

    # action待ちプレイヤーのリストをアクション順に整理、SB,BBのプレイヤーを取得
    shift_num = waiting_action_players.index(BTN_pos)       # BTNが何番目かを調べる
    waiting_action_players.rotate(-shift_num)               # BTNが頭にくるようにrotateする

    if len(waiting_action_players) != 2:
        # 非ヘッズアップ時
        SB_pos = waiting_action_players[1]                  # SBのシート番号を得る
        BB_pos = waiting_action_players[2]                  # BBのシート番号を得る
        waiting_action_players.rotate(-3)                   # 初めにアクションする人を先頭にする
    else:
        # ヘッズアップ時
        SB_pos = waiting_action_players[0]                  # ヘッズアップ時はBTN = SB。かつBTNが先にアクション
        BB_pos = waiting_action_players[1]                  # ヘッズアップ時はBTN+1 = BB

    # stackにanteとSB/BBを戻しておく
    for seat_no in waiting_action_players:
        if player_stack[seat_no] != 0:
            player_stack[seat_no] += ante

            if seat_no == SB_pos:
                 player_stack[seat_no] += SB
            elif seat_no == BB_pos:
                 player_stack[seat_no] += BB
        else:
            print("BB支払い以前でオールイン発生。例外処理要")
            raise Exception("End of movie")

    # 開始時のプレイヤーとそのチップ量一覧の出力
    for i in range(len(waiting_action_players)):
        seat_no = waiting_action_players[i]
        chip = player_stack[seat_no]

        # ex.Seat 1: Hero (24740 in chips)
        line = "Seat " + str(seat_no) + ": " + player_name_dict[seat_no] + " (" + str(chip) + " in chips)"
        output_txt.append(line)
        print(line)

    # anteの支払い
    player_betting = [0 for i in range(7)]                  # 各プレイヤーのそのラウンドでのBET額合計

    for i in range(len(waiting_action_players)):
        seat_no = waiting_action_players[i]

        if player_stack[seat_no] > ante:
            # 非オールイン時
            line = player_name_dict[seat_no] + ": posts the ante " + str(ante)
            player_betting[seat_no] = ante
            player_stack[seat_no] -= ante
        else:
            # オールイン時
            line = player_name_dict[seat_no] + ": posts the ante " + str(player_stack[seat_no]) + " and is all-in"
            player_betting[seat_no] = player_stack[seat_no]
            player_stack[seat_no] = 0
            is_allin[seat_no] = True
            allin_players.append(seat_no)
            waiting_action_players.remove(seat_no)

        output_txt.append(line)
        print(line)

    game_pot = convert_bet_to_pot(player_betting,is_folded)   # ゲーム毎のポット。[チップ量,[獲得権利のあるプレイヤーのリスト]]を格納する

    player_betting = [0 for i in range(7)]                  # 各プレイヤーのそのラウンドでのBET額合計

    # SBの支払い
    betting_list = list()    # (BET額,そのBETをしたプレイヤー)を格納していく

    if player_stack[SB_pos] > SB:
        # 非オールイン時
        line = player_name_dict[SB_pos] + ": posts small blind " + str(SB)
        betting_list.append([SB,SB_pos])
        player_betting[SB_pos] += SB
        player_stack[SB_pos] -= SB
    else:
        # オールイン時
        line = player_name_dict[SB_pos] + ": posts small blind " + player_stack[seat_no] + " and is all-in"
        betting_list.append([player_stack[SB_pos],SB_pos])
        player_betting[SB_pos] += player_stack[SB_pos]
        player_stack[SB_pos] = 0
        is_allin[SB_pos] = True
        allin_players.append(SB_pos)
        waiting_action_players.remove(SB_pos)

    output_txt.append(line)
    print(line)

    # BBの支払い
    if player_stack[BB_pos] > BB:
        # 非オールイン時
        line = player_name_dict[BB_pos] + ": posts small blind " + str(BB)
        betting_list.append([BB,BB_pos])
        player_betting[BB_pos] += BB
        player_stack[BB_pos] -= BB
    else:
        #オールイン時
        line = player_name_dict[BB_pos] + ": posts small blind " + player_stack[seat_no] + " and is all-in"
        betting_list.append([player_stack[BB_pos],BB_pos])
        player_betting[BB_pos] += player_stack[BB_pos]
        player_stack[BB_pos] = 0
        is_allin[BB_pos] = True
        allin_players.append(BB_pos)
        waiting_action_players.remove(BB_pos)

    output_txt.append(line)
    print(line)

    # Heroへの配布カード ... ex.*** HOLE CARDS ***  <BR> Dealt to Hero [2c 4c]
    line = "*** HOLE CARDS ***"
    output_txt.append(line)
    print(line)

    player_cards_left,player_cards_right = get_player_cards_info(img_rgb,1)
    dealt_card[1] = [player_cards_left,player_cards_right]
    line = "Dealt to " + player_name_dict[1] + " [" + player_cards_left + " " + player_cards_right + "]"
    output_txt.append(line)
    print(line)

    # 各ベッティングラウンドを進行する
    for betting_round in ["Before Flop","FLOP","TURN","RIVER","SHOW DOWN"]:
        if betting_round == "Before Flop":
            max_bet = BB
            board_card = list()
        else:
            max_bet = 0
            player_betting = [0 for i in range(7)]                  # 各プレイヤーのそのラウンドでのBET額合計
            line = "*** " + betting_round + " *** "

            # board cardの取得
            if betting_round == "FLOP":

                while True :
                    for i in range(skip_frames):
                        ret,img_rgb = cap.read()   #フレーム情報取得

                    time.sleep(sleep_time)

                    if ret == False:
                        #動画が終われば処理終了
                        raise Exception("End of movie")
                    else:
                        board_card3 = get_board_cards_info(img_rgb,3)

                        if board_card3 != -1:
                            board_card.append(get_board_cards_info(img_rgb,1))
                            board_card.append(get_board_cards_info(img_rgb,2))
                            board_card.append(get_board_cards_info(img_rgb,3))
                            line = "*** FLOP *** [" + board_card[0] + " " + board_card[1] + " " + board_card[2] +"]"
                            break


            elif betting_round == "TURN":
                while True :
                    for i in range(skip_frames):
                        ret,img_rgb = cap.read()   #フレーム情報取得

                    time.sleep(sleep_time)

                    if ret == False:
                        #動画が終われば処理終了
                        raise Exception("End of movie")
                    else:
                        board_card4 = get_board_cards_info(img_rgb,4)

                        if board_card4 != -1:
                            board_card.append(get_board_cards_info(img_rgb,4))
                            line = "*** TURN *** [" + board_card[0] + " " + board_card[1] + " " + board_card[2] + "] [" + board_card[3] + "]"
                            break

            elif betting_round == "RIVER":
                while True :
                    for i in range(skip_frames):
                        ret,img_rgb = cap.read()   #フレーム情報取得

                    time.sleep(sleep_time)

                    if ret == False:
                        #動画が終われば処理終了
                        raise Exception("End of movie")
                    else:
                        board_card5 = get_board_cards_info(img_rgb,5)

                        if board_card5 != -1:
                            board_card.append(get_board_cards_info(img_rgb,5))
                            line = "*** RIVER *** [" + board_card[0] + " " + board_card[1] + " " + board_card[2] + "] [" + board_card[3] + "] [" + board_card[4] + "]"
                            break

            output_txt.append(line)
            print(line)


        # アクション/ショーダウン順の整理
        if betting_round != "Before Flop" and betting_round != "SHOW DOWN":
            betting_list = list()                                               # (BET額,そのBETをしたプレイヤー)を格納していく
            waiting_action_players = list(action_finished_players)              # action待ちのプレイヤーの一覧
            action_finished_players = list()                                    # action済のプレイヤーの一覧

            # 残りプレイヤーをaction順になるようにする
            waiting_action_players.sort(reverse = False)
            shift = bisect.bisect_right(waiting_action_players,BTN_pos)         # ボタンの次のプレイヤーのindexを求める
            waiting_action_players = deque(waiting_action_players)              # dequeに変換
            waiting_action_players.rotate(-shift)                               # ボタンの次のプレイヤーが頭に来るようにrotate

        elif betting_round == "SHOW DOWN":
            if len(action_finished_players) != 0:
                aggressor_seat_no = action_finished_players[0]
            else:
                aggressor_seat_no = allin_players[0]

            showdown_players = list(action_finished_players) + allin_players    # show downする可能性のあるプレイヤーのリスト
            shift_num = showdown_players.index(aggressor_seat_no)               # アグレッサーが何番目かを調べる
            showdown_players = deque(showdown_players)
            showdown_players.rotate(-shift_num)                                 # アグレッサーが頭にくるようにrotateする


        # アクション/ショーダウンの進行
        if betting_round == "SHOW DOWN":
            # ショーダウン時
            hand_rank = ["*" for _ in range(7)]
            hand_rank_in_str = ["*" for _ in range(7)]  # 文字列で表したハンド
            hand_rank_list = list()     # (ハンドランク,シートNo)を格納するリスト

            # 誰かにWINNER表示が出るのを待つ
            break_flag = False
            while True :
                for i in range(skip_frames):
                    ret,img_rgb = cap.read()   #フレーム情報取得

                time.sleep(sleep_time)

                if ret == False:
                    #動画が終われば処理終了
                    raise Exception("End of movie")
                else:
                    for seat_no in showdown_players:
                        if is_winner(img_rgb,seat_no):
                            break_flag = True
                            break

                    if break_flag:
                        break

            for seat_no in showdown_players:
                # ショーダウンするプレイヤーのカードを取得
                dealt_cards_left,dealt_cards_right = get_player_cards_info(img_rgb,seat_no)
                dealt_card[seat_no] = [dealt_cards_left,dealt_cards_right]

                # 役判定
                if dealt_card[seat_no][0][0] == "0" or dealt_card[seat_no][1][0] == "0":
                    hand_rank[seat_no] = ["muck","muck"]
                else:
                    hand_rank[seat_no] = get_strongest_hand_rank(dealt_card[seat_no],board_card)
                    hand_rank_list.append((hand_rank[seat_no][0],hand_rank[seat_no][1],seat_no))

            hand_rank_list.sort(reverse=True)               # 左の方がハンドランク強いプレイヤーの並びになった
            winning = get_winning(game_pot,hand_rank_list)  # 各プレイヤーが獲得できる金額を得る

            # ショーしたハンドと出来た役
            for seat_no in showdown_players:
                hand,cards_combi = hand_rank[seat_no]

                if hand != "muck":
                    hand_str = conv_hand_to_str(hand,cards_combi)
                    line = player_name_dict[seat_no] + ": shows [" + dealt_card[seat_no][0] + " " + dealt_card[seat_no][1] + "] (" + hand_str + ")"
                    hand_rank[seat_no][0] = hand_str
                else:
                     line = player_name_dict[seat_no] + ": mucks hand"

                output_txt.append(line)
                print(line)

            # 獲得したポット
            for seat_no in showdown_players:
                if winning[seat_no][0] != 0:
                    line = player_name_dict[seat_no] + " collected " + str(winning[seat_no][0]) + " from pot"
                    output_txt.append(line)
                    print(line)


        elif betting_round != "SHOW DOWN":
            # 各ベッティングラウンド
            is_first_bet = True    # 初betか否か
            before_action = [None for i in range(7)]    # 直前のアクション
            before_betting = [None for i in range(7)]   # 直前のベッティング額

            if len(waiting_action_players) == 1 and len(allin_players) != 0:
                    # all-in勝負なのでBETは行われない
                    action_finished_players = waiting_action_players
                    pass
            else:
                # アクションする権利のあるプレイヤーがいなくなるまで回し続ける
                while len(waiting_action_players) != 0:

                    # 手前のプレイヤーが全員フォールドしていないかの確認
                    if len(waiting_action_players) == 1 and len(allin_players) == 0:
                        for seat_no in action_finished_players:
                            if is_folded[seat_no] == betting_round:
                                pass
                            else:
                                break
                        else:
                            # 手前のプレイヤーが全員フォールドしていた場合はアクションを拾う必要なく終了
                            action_finished_players.append(waiting_action_players.pop())
                            break

                    # アクションを拾う
                    seat_no = waiting_action_players.popleft()

                    while True :
                        for i in range(skip_frames):
                            ret,img_rgb = cap.read()   #フレーム情報取得

                        time.sleep(sleep_time)

                        if ret == False:
                            #動画が終われば処理終了
                            raise Exception("End of movie")
                        else:
                            player_action = get_player_action(img_rgb,seat_no,BB)

                            if player_action["action"] != None:
                                # 何らかのアクションを拾えて、以前の状態と異なっていたら(オールイン時などのときの重複取得防止)、アクションを正しく拾えたと判断して抜ける
                                if player_action["action"] != before_action[seat_no] or player_action["bet_size"] != before_betting[seat_no]:
                                    before_action[seat_no] = player_action["action"]
                                    before_betting[seat_no] = player_action["bet_size"]
                                    break


                    # アクションに応じて操作
                    if player_action["action"] == "Fold":
                        is_folded[seat_no] == betting_round
                        line = player_name_dict[seat_no] + ": folds"
                        is_folded[seat_no] = betting_round

                    elif player_action["action"] == "Check":
                        betting_list.append([0,seat_no])
                        action_finished_players.append(seat_no)
                        line = player_name_dict[seat_no] + ": checks"

                    elif player_action["action"] == "Call":
                        betting_list.append([max_bet,seat_no])
                        player_betting[seat_no] = max_bet
                        player_stack[seat_no] -= player_action["bet_size"] - player_betting[seat_no]
                        action_finished_players.append(seat_no)
                        line = player_name_dict[seat_no] + ": calls " + str(max_bet)

                    elif player_action["action"] == "Bet" or player_action["action"] == "Raise":
                        betting_list.append([player_action["bet_size"],seat_no])
                        player_stack[seat_no] -= player_action["bet_size"] - player_betting[seat_no]
                        waiting_action_players.extend(action_finished_players)
                        action_finished_players = [seat_no]

                        if player_action["action"] == "Bet":
                            line = player_name_dict[seat_no] + ": bets " + str(player_action["bet_size"])
                            is_first_bet = False
                        elif player_action["action"] == "Raise":
                            line = player_name_dict[seat_no] + ": raises " + str(player_action["bet_size"] - player_betting[seat_no]) + " to " + str(player_action["bet_size"])
                            is_first_bet = False

                        max_bet = player_action["bet_size"]
                        player_betting[seat_no] = player_action["bet_size"]

                    elif player_action["action"] == "Allin":
                        betting_list.append([player_action["bet_size"],seat_no])
                        is_allin[seat_no] = True
                        allin_players.append(seat_no)

                        if player_action["bet_size"] > max_bet:
                            waiting_action_players.extend(action_finished_players)
                            action_finished_players = list()

                            if is_first_bet:
                                line = player_name_dict[seat_no] + ": bets " + str(player_action["bet_size"]) + " and is all-in"
                                is_first_bet = False
                            else:
                                line = player_name_dict[seat_no] + ": raises " + str(player_action["bet_size"] -  player_betting[seat_no]) + " to " + str(player_action["bet_size"]) + " and is all-in"

                            max_bet = player_action["bet_size"]
                        else:
                            line = player_name_dict[seat_no] + ": calls " + str(player_action["bet_size"]) + " and is all-in"

                        player_betting[seat_no] = player_action["bet_size"]
                        player_stack[seat_no] = 0

                    output_txt.append(line)
                    print(line)


                # 全員のアクションが終わったのでpotを整理
                this_round_pot = convert_bet_to_pot(player_betting,is_folded)

                # 1人のみ獲得権利があるポットはUncalledとして戻す
                if len(this_round_pot) != 0 and len(this_round_pot[-1][1]) == 1:
                    bet,seat_no_list = this_round_pot.pop(-1)
                    seat_no = seat_no_list[0]

                    # ex. Uncalled bet (9975) returned to ElT007
                    line = "Uncalled bet (" + str(bet) + ") retured to " + player_name_dict[seat_no]
                    output_txt.append(line)
                    print(line)

                if len(this_round_pot)!=0 and game_pot[-1][1] == this_round_pot[0][1]:
                    # 獲得権利があるプレイヤーが同一のポットをまとめる
                    game_pot[-1][0] += this_round_pot[0][0]
                    if len(this_round_pot) != 1:
                        # 獲得権利があるプレイヤーが同一でないポットがあればサイドとして追加
                        game_pot.extend(this_round_pot[1:])
                else:
                    game_pot.extend(this_round_pot)


                # 次のbetting roundに進むか判定
                if (len(action_finished_players) == 1 and len(allin_players) == 0) or (len(action_finished_players) == 0 and len(allin_players) == 1):
                    # 進まない場合は、potを勝者に1人に全渡ししてゲーム終了する

                    # ポット量の計算と勝者取得
                    remain_player = action_finished_players + allin_players
                    seat_no = remain_player[0]
                    total_pot = sum([game_pot[i][0] for i in range(len(game_pot))])
                    game_pot = [[total_pot,[seat_no]]]
                    acquired[seat_no] = total_pot

                    # ex. Hero collected $3.65 from pot
                    line = player_name_dict[seat_no] + " collected " + str(total_pot) + " from pot"
                    output_txt.append(line)
                    print(line)

                    # ex. Hero: doesn't show hand
                    line = player_name_dict[seat_no] + ": doesn't show hand "
                    output_txt.append(line)
                    print(line)

                    break


    # SUMMARY の出力
    line = "*** SUMMARY ***"
    output_txt.append(line)
    print(line)

    if len(game_pot) == 1:
        # ex. Total pot 18010 | Rake 0
        line = "Total pot " + str(game_pot[0][0]) + " | Rake 0"
        output_txt.append(line)
        print(line)
    else:
        # Total pot 74303 Main pot 34212. Side pot-1 29073. Side pot-2 11018. | Rake 0
        total_pot = sum([game_pot[i][0] for i in range(len(game_pot))])
        line = "Total pot " + str(total_pot) + " Main pot " + str(game_pot[0][0]) + ". "

        for i in range(1,len(game_pot)):
            line += "Side pot-" + str(i) + " " + str(game_pot[i][0]) + ". "

        line += "| Rake 0"

    if len(board_card) != 0:
        # ex.Board [Tc Jd 6c 7c]
        line = "Board [" + ' '.join(map(str,board_card)) + "]"
        output_txt.append(line)
        print(line)

    for seat_no in range(1,7):
        if is_folded[seat_no] != "Finished":
            line = "Seat " + str(seat_no) + ": " + player_name_dict[seat_no] + " "

            if seat_no == SB_pos:
                line += "(small blind) "
            elif seat_no == BB_pos:
                line += "(big blind) "

            if seat_no == BTN_pos:
                line += "(button) "

            if is_folded[seat_no] != False:
                if is_folded[seat_no] == "Before Flop":
                    line += "folded before Flop"
                else:
                    line += "folded on the " + is_folded[seat_no].capitalize()

                if seat_no == SB_pos and seat_no == BB_pos:
                    line += " (didn't bet)"
            else:
                if betting_round != "SHOW DOWN":
                    line += "collected (" + str(acquired[seat_no]) + ")"
                else:
                    if winning[seat_no][0] == 0:
                        if hand_rank[seat_no][0] == "muck":
                            line += "mucked"
                        else:
                            line += "showed [" + dealt_card[seat_no][0] + " " + dealt_card[seat_no][1] + "] and lost with " + hand_rank[seat_no][0]
                    else:
                        line += "showed [" + dealt_card[seat_no][0] + " " + dealt_card[seat_no][1] + "] and won (" + str(winning[seat_no][0]) + ") with " + hand_rank[seat_no][0]

            output_txt.append(line)
            print(line)

    # ●ゲーム終了時の処理

    # ハンドヒストリーに改行を開ける
    for i in range(3):
        line = ""
        output_txt.append(line)
        print(line)

    # 次のハンドを待つ
    # プレイヤーに手札が配られていない状態まで待機
    while True :

        for i in range(skip_frames):
            ret,img_rgb = cap.read()   #フレーム情報取得

        time.sleep(sleep_time)

        if ret == False:
            #動画が終われば処理終了
            raise Exception("End of movie")
        else:
            if is_card_dealt_to_hero(img_rgb) == "No":
                break

    # プレイヤーに手札が配られるまで待機
    while True :
        #print("Frame: "+ str(i))

        for i in range(skip_frames):
            ret,img_rgb = cap.read()   #フレーム情報取得

        time.sleep(sleep_time)

        if ret == False:
            #動画が終われば処理終了
            raise Exception("End of movie")
        else:
            if is_card_dealt_to_hero(img_rgb) == "Yes(Not fold)":
                # 他プレイヤーへのカード配布されるのを待つ分、フレームを捨てる
                for i in range(skip_frames * 2):
                    ret,img_rgb = cap.read()   #フレーム情報取得
                    if ret == False:
                        #動画が終われば処理終了
                        raise Exception("End of movie")
                break
