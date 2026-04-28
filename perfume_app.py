import streamlit as st
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 1. 镜像加速设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 补全的 50 条香水数据
FULL_PERFUME_DATA = [
    {"id": "m1", "name": "Dior Sauvage", "gender": "男士", "doc": "霸道型男 荷尔蒙 金属感 粗犷 干净 辛辣柑橘 龙涎香 旷野"},
    {"id": "m2", "name": "Bleu de Chanel", "gender": "男士", "doc": "稳重蓝调 职场精英 安心感 雪松 明亮清新 稳重不沉重"},
    {"id": "m3", "name": "Creed Aventus", "gender": "男士", "doc": "征服者 烟熏菠萝 卓越 拿破仑之水 成熟成功人士"},
    {"id": "m4", "name": "Acqua di Gio", "gender": "男士", "doc": "清爽水生 白衬衫 海风 柑橘海水 透明感 少年气"},
    {"id": "m5", "name": "Hermes Terre d Hermes", "gender": "男士", "doc": "大地矿石 雨后泥土 包容 葡萄柚 燧石 香根草 沉着 哲学"},
    {"id": "m6", "name": "Tom Ford Grey Vetiver", "gender": "男士", "doc": "职场教科书 洁净 专业 香根草核心 高效率 冷色调 商务"},
    {"id": "m7", "name": "Versace Eros", "gender": "男士", "doc": "派对之王 挑逗 激情 薄荷 青苹果 香草 夜店 高甜度 多巴胺"},
    {"id": "m8", "name": "YSL Y Eau de Parfum", "gender": "男士", "doc": "奋斗者 锐意 鼠尾草 职场上升期年轻人 活力"},
    {"id": "m9", "name": "Maison Margiela Jazz Club", "gender": "男士", "doc": "微醺 烟草 朗姆酒 昏暗酒吧 爵士乐 故事感 忧郁 复古"},
    {"id": "m10", "name": "Le Labo Santal 33", "gender": "男士", "doc": "都市文青 书卷气 疏离 皮革檀香 纸莎草 艺术感 中性"},
    {"id": "m11", "name": "Valentino Uomo Born in Roma", "gender": "男士", "doc": "浪漫叛逆 精致 现代 生姜 烟熏香根草 意式潮流 玩世不恭"},
    {"id": "m12", "name": "Byredo Bibliotheque", "gender": "男士", "doc": "静谧图书馆 怀旧 沉思 旧书页 皮革封面 慢节奏 知识分子"},
    {"id": "m13", "name": "Prada L Homme", "gender": "男士", "doc": "极佳皂感 秩序 洁净 鸢尾花 刚洗完澡的白衬衫 温柔绅士"},
    {"id": "m14", "name": "Azzaro Wanted by Night", "gender": "男士", "doc": "温暖性感 肉桂 深沉 橘子 适合秋冬 磁场吸引力"},
    {"id": "m15", "name": "Viktor&Rolf Spicebomb", "gender": "男士", "doc": "辛辣炸弹 爆发 阳刚 存在感极强 炽热 肌肉感"},
    {"id": "m16", "name": "Dior Homme Intense", "gender": "男士", "doc": "贵族绅士 粉质鸢尾 克制 柔软但有骨架 儒雅精致"},
    {"id": "m17", "name": "Acqua di Parma Colonia", "gender": "男士", "doc": "意式阳光 经典 明亮 柑橘古龙水 地中海夏日街道"},
    {"id": "m18", "name": "Burberry Hero", "gender": "男士", "doc": "坚韧 英勇 雪松 杜松子 皮革 安息香 荒野马背 勇气"},
    {"id": "m19", "name": "Paco Rabanne 1 Million", "gender": "男士", "doc": "金砖 财富 社交自信 甜美肉桂 派对 浮夸 成功"},
    {"id": "m20", "name": "Loewe 001 Man", "gender": "男士", "doc": "事后清晨 暧昧 随性 木质调 阳光洒在床单上的自然感"},
    {"id": "m21", "name": "Aesop Hwyl", "gender": "男士", "doc": "禅意 古老森林 冥想 烟熏 柏树 日本森林 孤独思考"},
    {"id": "m22", "name": "Malin+Goetz Dark Rum", "gender": "男士", "doc": "微甜朗姆 诱惑 独特性感 黑朗姆 皮革 深溺夜色"},
    {"id": "m23", "name": "Issey Miyake L Eau d Issey", "gender": "男士", "doc": "极简水生 纯净 禅 清凉山泉水 宁静"},
    {"id": "m24", "name": "Jo Malone Cypress & Grapevine", "gender": "男士", "doc": "丝柏森林 可靠 稳重 葡萄藤 深秋感 不带侵略性的绅士"},
    {"id": "m25", "name": "Cremo Italian Bergamot", "gender": "男士", "doc": "平价黑马 务实 清新 佛手柑 简洁 日常活力"},
    {"id": "w1", "name": "Chanel Coco Mademoiselle", "gender": "女士", "doc": "独立 坚定 古灵精怪 职场 活力权利平衡 柑橘 玫瑰 广藿香 柔美摩登"},
    {"id": "w2", "name": "Dior J adore Parfum d Eau", "gender": "女士", "doc": "优雅 柔软 女性化 透明感 橙花 桑巴茉莉 纯净白花原香 无酒精水基"},
    {"id": "w3", "name": "Glossier You", "gender": "女士", "doc": "贴肤 自我拥抱 亲密 干净 降龙涎香醚 鸢尾花 活在当下 皮肤温暖感"},
    {"id": "w4", "name": "Jo Malone English Pear & Freesia", "gender": "女士", "doc": "秋日果园 知性 轻盈 英国庄园 威廉梨 小苍兰 凉爽感"},
    {"id": "w5", "name": "YSL Libre", "gender": "女士", "doc": "自由 刚柔并济 冷艳 张扬 薰衣草 橙花 阳刚与妩媚的张力"},
    {"id": "w6", "name": "Gucci Bloom", "gender": "女士", "doc": "繁盛花园 浪漫 白花 绿意 晚香玉 茉莉 鸢尾根 内敛凉意"},
    {"id": "w7", "name": "Maison Margiela Lazy Sunday Morning", "gender": "女士", "doc": "慵懒 周末 白衬衫 洁净 阳光晒过床单 皂感 悠然自得"},
    {"id": "w8", "name": "Tom Ford Black Orchid", "gender": "女士", "doc": "黑暗 欲望 奢华 侵略性 松露 巧克力 广藿香 深邃独一无二"},
    {"id": "w9", "name": "Narciso Rodriguez For Her", "gender": "女士", "doc": "纯欲 神秘 内敛 深度 核心麝香 橙花 桂花 原始吸引力"},
    {"id": "w10", "name": "Byredo Gypsy Water", "gender": "女士", "doc": "灵性 自然 流浪 梦幻 杜松子 焚香 柠檬 香草 森林篝火自由"},
    {"id": "w11", "name": "Diptyque Do Son", "gender": "女士", "doc": "海边 思念 电影感 宁静 晚香玉 橙花 玫瑰 湿润微咸海风"},
    {"id": "w12", "name": "Parfums de Marly Delina", "gender": "女士", "doc": "贵气 甜美 公主 派对 土耳其玫瑰 牡丹 荔枝 华丽存在感"},
    {"id": "w13", "name": "Le Labo Another 13", "gender": "女士", "doc": "高冷 极简 城市 现代幽灵 降龙涎香醚 麝香 金属般的温暖"},
    {"id": "w14", "name": "Marc Jacobs Daisy", "gender": "女士", "doc": "喜悦 纯真 阳光 野性 香蕉花 户外自然 俏皮无忧无虑"},
    {"id": "w15", "name": "Prada Paradoxe", "gender": "女士", "doc": "矛盾 创新 多面 先锋 茉莉 琥珀 白麝香 突破界限对比感"},
    {"id": "w16", "name": "Kilian Love Don t Be Shy", "gender": "女士", "doc": "渴望 童真 甜美 沉溺 橙花 棉花糖 焦糖 香草 美食调诱惑"},
    {"id": "w17", "name": "Billie Eilish Eilish No 2", "gender": "女士", "doc": "阴郁 野性 冷静 雨后感 湿木头 泥土 雨后人行道水泥味"},
    {"id": "w18", "name": "Sol de Janeiro Cheirosa 62", "gender": "女士", "doc": "快乐 假期 咸味焦糖 火热 阳光沙滩 咸甜交织 能量"},
    {"id": "w19", "name": "Celine Parade", "gender": "女士", "doc": "复古 正式 距离感 克制 佛手柑 橙花 香根草 橡木苔 贵族粉感"},
    {"id": "w20", "name": "Snif Heal the Way", "gender": "女士", "doc": "治愈 安全感 拥抱 深夜 檀香 奶油 毛茸茸的慰藉"},
    {"id": "w21", "name": "Diptyque Orpheon", "gender": "女士", "doc": "故事 微醺 沙龙感 复古 烟草 木质吧台 鸢尾 粉感复古花香"},
    {"id": "w22", "name": "Burberry Her", "gender": "女士", "doc": "活力 少女 英伦都市 节奏 黑加仑 蓝莓 伦敦城市繁华果断"},
    {"id": "w23", "name": "Kayali Vanilla 28", "gender": "女士", "doc": "包裹 甜蜜 稳定 舒适 醇厚香草 红糖 冬日壁炉 沉静暖香"},
    {"id": "w24", "name": "Hermes Twilly d Hermes", "gender": "女士", "doc": "叛逆 趣味 色彩 古灵精怪 生姜 晚香玉 非传统个性"},
    {"id": "w25", "name": "Aerin Mediterranean Honeysuckle", "gender": "女士", "doc": "清爽 透明 逃离 地中海 忍冬 葡萄柚 蔚蓝海岸退隐"}
]

@st.cache_resource
def init_engine():
    client = chromadb.PersistentClient(path="./perfume_vector_db")
    emb_fn = SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    collection = client.get_or_create_collection(name="perfume_collection", embedding_function=emb_fn)
    
    # 如果库空了（或者路径不对），自动重录
    if collection.count() == 0:
        collection.add(
            ids=[p["id"] for p in FULL_PERFUME_DATA],
            documents=[p["doc"] for p in FULL_PERFUME_DATA],
            metadatas=[{"name": p["name"], "gender": p["gender"]} for p in FULL_PERFUME_DATA]
        )
    return collection

st.title("🔮 气味占卜：寻找你的灵魂香氛")
engine = init_engine()

user_input = st.text_area("输入你此刻的心境：", height=100)

if st.button("开始占卜"):
    if user_input.strip():
        results = engine.query(query_texts=[user_input], n_results=1)
        if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
            res = results['metadatas'][0][0]
            st.balloons()
            st.subheader(f"结果：【{res['name']}】 ({res['gender']})")
    else:
        st.warning("请输入文字。")
