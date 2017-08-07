#/usr/bin/python3
# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

# commit hash
# 9782c18

import numpy as np

from hyperparams import Hyperparams as hp
import re
import os
import csv
import codecs
from itertools import chain, repeat


def load_vocab():
    if hp.data_set == 'bible':
        return load_vocab_en()
    elif hp.data_set == 'atr503':
        return load_vocab_ja_hiragana()
    elif hp.data_set == 'siwis':
        return load_vocab_fr()
    else:
        raise ValueError('unknown data set')


def load_vocab_en():
    vocab = "EG ',.abcdefghijklmnopqrstuvwxyz"  # E: Empty. ignore G
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char


def load_vocab_fr():
    vocab = " ',.abcdefghijklmnopqrstuvwxyzàâçèéêëîïôùûü"  # E: Empty. ignore G
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char


def load_vocab_ja_hiragana():
    vocab = "、。ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわをんー"
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char


def load_vocab_ja():
    vocab = "、。々ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわをんァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヲンヴヶー一丁七万三上下不与世両並中主久乗九乱乳乾了予争事二互五交京人今介仏仕他付代以件任企伏休会伝伴伸似位住佐体何余作併使供依価侵便係保信修俺倉個倍倒候倫偉健側偶偽備傷働僕償優元兄充先光免児入全八公六具内円再写冬冷凍几凡処出分切刊列初別利制券刺則前剤剥割劇力助努労効勇勉動務勝勢勤包化匠匹医十千午半協単印危却卵原厳去参及友反収叔取受口古叩台史右司各合同名吐向君否含吸吹呆告周味呼命和品員唯問善喜嘆器四回因困囲固国園土圧在地坂均坊型垢埋城域堂報場塀塁塵境墓増壁壊士声売変夏夕外多夜夢大天太夫失奇奈奏奮女好妙妬妻始姿威娘婚婦嫌嬉子字存季学宅宇守安完宙定宝実客室宮害家宿寄密富寒寝察寸寺寿専射将尋導小少尺局屈届屋属層山岩岬岸島崩川工巧差己巻市布希師席帰帳常帽幌干平年幸幼庁広床底店府度庫庭康延建弁弊式引弟弱張強弾当形彩影彼往待後徐得御復微徴徹心必忍志応忠念怒怖思怠急性怪恐息恵悔患悩悪悲情惑想意愛感態慌慢慮慰憐憩懐懸懺成我戦戻房所扇扉手打払扱扶承技抑抒投折抜押拍招拠拡拷持指挙振捕捨授掌排掘掛採探接描揚握揮援揺撃撮擦支改攻放政故救敗教散数整敷文料新方施旅旋族日旧早昇明易昔星春昭昼時晩普景晴暑暖暮暴曜曲更書替最月有服望朝期木末本札材村束条来杯東板枚果枝枠染柔柱査栄栓校根格案械棋植検業極楽構様標模横機欠次欧欲歌歓止正歩歯歳歴死残殖段殻母毎毒比毛民気水氷永氾求汗汚決沈没河油治沼沿況泊法波泣泥注洋洗活流浄浅浜浮海浸消涙淋淡深淵混済減渡湖湯満源準溜滅滑漁漏演潜潤激濃濫火灯炎点為無焦然焼照熊熱燃燐爆父片版牛物牽状狙狩独狭猟猫獅玄率玉王珍現球理璧甘生産用由男町界畑留番異畳疲病症痛痢痴療癖発登白百的皮盛盤目直相省真眠眺眼着睡瞬瞳知短石砂研破碑示社祖神票祭秀私秋秘称移程税種稿積穴究空突窒窓立章端競竹笑笛符筆筒答算管節築米粉粧精糸系糾約紅紋純紙級紛素細終組経結給絵絶絹継続綴綿緊緒線編練縁縦縫縮繁繊繕纏罅置署羊美群義羽翌習翻老考者耐耳聖聞職肉肋肌肘肥肩育胃背胴胸能脅脈脚脱脳腐腕腫腸腹膜膝自致興舗舞舟航般舶船艇良色花芸苑苗若苦英茶草菜落葉蔵薄薬虚虜虫融血衆行術街衛衣表袋袖裂装裏裕補製裾複襞襟要覆見規視覚親観角解触言計訓記訪設許訳診証評試詰話誌認語説読誰課調談論諸謙講識警議譲護豆豊豚象負財貧販責買貿賀賃資賑賛質赤走赴起超越足距跡路踊踏躍身車軍軒転輪輸辛辞農辺込迎近返追送逃逆透途通速造連週進遂遅遊運遍過道達違遠適選遺避邪部郵郷都配酒酷醜里重野量金鈍鉄鉛鉢銘錨鍵長門閉開間関闇防降限院除陥陸隊階隔際障雄集雑離難雨雪雰雲電需震霧露青静非面鞄音響頑頭頼題顔願類顧風飛食飯飲養餐餓館首馬駄駅駐騒験骨高髪鮮鯨鰻鳥鳴麗麦黄黒黙鼻齢"
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char


def load_phone_ja():
    phones = [
        'A', 'I', 'N', 'O', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 'e',
        'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny',
        'o', 'p', 'pau', 'py', 'r', 'ry', 's', 'sh', 'sil', 't', 'ts', 'u',
        'w', 'y', 'z'
    ]
    phone2idx = {phone: idx for idx, phone in enumerate(phones)}
    idx2phone = {idx: phone for idx, phone in enumerate(phones)}
    return phone2idx, idx2phone


def create_train_data():
    if hp.data_set == 'bible':
        return create_train_data_bible()
    elif hp.data_set == 'atr503':
        sound_files, texts_mixed, texts_kana, phones = create_train_data_atr503(
        )
        return sound_files, texts_kana
    elif hp.data_set == 'siwis':
        return create_train_data_siwis()
    else:
        raise ValueError('unknown data set')


def create_dual_source_train_data():
    if hp.data_set == 'bible_siwis':
        return create_bible_siwis_train_data()
    if hp.data_set == 'atr503_dual':
        sound_files, texts_mixed, texts_kana, texts_phone = create_train_data_atr503(
        )
        return sound_files, texts_kana, texts_phone
    else:
        raise ValueError('unknown data set')


def create_bible_siwis_train_data():
    bible_sound, bible_text = create_train_data_bible()
    siwis_sound, siwis_text = create_train_data_siwis()
    bible_text = bible_text + list(repeat("", len(siwis_text)))
    siwis_text = list(repeat("", len(bible_text))) + siwis_text
    sounds = bible_sound + siwis_sound
    return sounds, bible_text, siwis_text


def create_train_data_bible():
    # Load vocabulary
    char2idx, idx2char = load_vocab_en()

    texts, sound_files = [], []
    reader = csv.reader(codecs.open(hp.bible_text_file, 'rb', 'utf-8'))
    for row in reader:
        sound_fname, text, duration = row
        if hp.reverse_input:
            text = text[::-1]
        sound_file = hp.bible_sound_fpath + "/" + sound_fname + ".wav"
        text = re.sub(r"[^ a-z.',]", "", text.strip().lower())

        if hp.min_len <= len(text) <= hp.max_len:
            texts.append(
                np.array([char2idx[char]
                          for char in text], np.int32).tostring())
            sound_files.append(sound_file)

    return sound_files, texts


def create_train_data_siwis():
    # Load vocabulary
    char2idx, idx2char = load_vocab_fr()

    texts, sound_files = [], []
    reader = csv.reader(codecs.open(hp.siwis_text_file, 'rb', 'utf-8'))
    for row in reader:
        sound_fname, text = row
        if hp.reverse_input:
            text = text[::-1]
        sound_file = hp.siwis_sound_fpath + "/" + sound_fname + ".wav"
        text = re.sub(r"[^ a-z.',çéâêîôûàèùëïü]", "", text.strip().lower())

        if hp.min_len <= len(text) <= hp.max_len:
            texts.append(
                np.array([char2idx[char]
                          for char in text], np.int32).tostring())
            sound_files.append(sound_file)

    return sound_files, texts


def create_train_data_atr503():
    # Load vocabulary
    char2idx_ja, idx2char_ja = load_vocab_ja()
    char2idx_kana, idx2char_kana = load_vocab_ja_hiragana()
    phone2idx, idx2phone = load_phone_ja()

    texts_mixed, texts_kana, texts_phone, sound_files = [], [], [], []
    reader = csv.reader(codecs.open(hp.atr503_text_file, 'rb', 'utf-8'))
    for row in reader:
        sound_index, text_mixed, text_hiragana, phones = row
        phones = phones.split(' ')
        if hp.reverse_input:
            text_mixed, text_hiragana, phones = text_mixed[::
                                                           -1], text_hiragana[::
                                                                              -1], phones[::
                                                                                          -1]

        sound_fname = "nitech_jp_atr503_m001_" + sound_index
        sound_file = hp.atr503_sound_fpath + "/" + sound_fname + ".wav"

        if hp.min_len <= len(text_hiragana) <= hp.max_len:
            texts_mixed.append(
                np.array([char2idx_ja[char]
                          for char in text_mixed], np.int32).tostring())

            texts_kana.append(
                np.array([char2idx_kana[char]
                          for char in text_hiragana], np.int32).tostring())

            texts_phone.append(
                np.array([phone2idx[phone]
                          for phone in phones], np.int32).tostring())

            sound_files.append(sound_file)

            assert len(texts_mixed) == len(texts_kana)
            assert len(texts_kana) == len(texts_phone)
            assert len(texts_phone) == len(sound_files)

    return sound_files, texts_mixed, texts_kana, texts_phone


def load_train_data():
    """We train on the whole data but the last num_samples."""
    sound_files, texts = create_train_data()
    if hp.sanity_check:  # We use a single mini-batch for training to overfit it.
        texts, sound_files = texts[:hp.
                                   batch_size] * 1000, sound_files[:hp.
                                                                   batch_size] * 1000
    else:
        texts, sound_files = texts[:-hp.num_samples], sound_files[:-hp.
                                                                  num_samples]
    return texts, sound_files


def load_dual_source_train_data():
    """We train on the whole data but the last num_samples."""
    sound_files, texts1, texts2 = create_dual_source_train_data()
    if hp.sanity_check:  # We use a single mini-batch for training to overfit it.
        # ToDo: mix texts1 and texts2
        sound_files, texts1, texts2 = sound_files[:hp.
                                                  batch_size] * 1000, texts1[:
                                                                             hp.
                                                                             batch_size] * 1000, texts2[:
                                                                                                        hp.
                                                                                                        batch_size] * 1000
    else:
        # ToDo: exclude samples
        sound_files, texts1, texts2 = sound_files[:-hp.
                                                  num_samples], texts1[:-hp.
                                                                       num_samples], texts2[:
                                                                                            -hp.
                                                                                            num_samples]
        pass

    return texts1, texts2, sound_files


def load_eval_data():
    """We evaluate on the last num_samples."""
    _, texts = create_train_data()
    if hp.sanity_check:  # We generate samples for the same texts as the ones we've used for training.
        texts = texts[:hp.batch_size]
    else:
        texts = texts[-hp.num_samples:]

    X = np.zeros(shape=[len(texts), hp.max_len], dtype=np.int32)
    for i, text in enumerate(texts):
        _text = np.fromstring(text, np.int32)  # byte to int
        X[i, :len(_text)] = _text

    return X


def load_dual_source_eval_data():
    """We evaluate on the last num_samples."""
    _, texts1, texts2 = create_dual_source_train_data()
    if hp.sanity_check:  # We generate samples for the same texts as the ones we've used for training.
        texts1, texts2 = texts1[:hp.batch_size], texts2[:hp.batch_size]
    else:
        texts1, texts2 = texts1[-hp.num_samples:], texts2[-hp.num_samples:]

    X1 = np.zeros(shape=[len(texts1), hp.max_len], dtype=np.int32)
    X2 = np.zeros(shape=[len(texts2), hp.max_len], dtype=np.int32)

    for i, text in enumerate(texts1):
        _text = np.fromstring(text, np.int32)  # byte to int
        X1[i, :len(_text)] = _text

    for i, text in enumerate(texts2):
        _text = np.fromstring(text, np.int32)  # byte to int
        X2[i, :len(_text)] = _text

    return X1, X2