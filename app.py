import base64
import io
import itertools
import math
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Експертне голосування", layout="wide")

OBJECTS = [
    "Sirius", "Black Hole", "Mercury", "Venus", "Andromeda",
    "Earth", "Mars", "Wormhole", "Jupiter", "Saturn",
    "Uranus", "Neptune", "Pluto", "Moon", "Europa",
    "Titan", "Milky Way", "Callisto", "Sun", "Comet",
]
EXPERTS = [
    "Вiка", "Анна", "Іван", "Ромчик", "Анастасiя",
    "Лiза", "Валерiя", "Лера", "Даша", "Настя",
    "Максим", "Веронiка", "Вiкторiя", "Дарина", "Марина",
    "Anastasiia", "Михайло", "Дарiя", "хтось", "Оскар",
    "Викладач",
]
HEURISTICS = {
    "E1": "Об'єкт обирався 1 раз на 3-му місці",
    "E2": "Об'єкт обирався 1 раз на 2-му місці",
    "E3": "Об'єкт обирався 1 раз  на 1-му місці",
    "E4": "Об'єкт обирався 2 рази на 3-му місці",
    "E5": "Об'єкт обирався 2 рази, один раз на 3-му і один раз на 2-му місці",
    "E6": "Сума балів <= 3",
    "E7": "Об'єкт жодного разу не обирався на 1-му місці",
}
VOTES_FILE = "votes.csv"
H_VOTES_FILE = "heuristic_votes.csv"
ADMIN_PASSWORD = "admin123"
SEED_H_VOTES = [
    ("Вiка","E7","E6","E1"),("Анна","E6","E7","E2"),("Іван","E7","E1","E4"),
    ("Ромчик","E6","E4","E5"),("Анастасiя","E7","E5","E6"),("Лiза","E1","E6","E7"),
    ("Валерiя","E6","E7","E3"),("Лера","E7","E2","E6"),("Даша","E6","E1","E7"),
    ("Настя","E7","E6","E4"),("Максим","E6","E5","E7"),("Веронiка","E7","E6","E1"),
    ("Вiкторiя","E6","E7","E5"),("Дарина","E7","E4","E6"),("Марина","E6","E7","E2"),
    ("Anastasiia","E7","E6","E3"),("Михайло","E6","E1","E7"),("Дарiя","E7","E6","E5"),
    ("хтось","E6","E7","E4"),("Оскар","E7","E5","E6"),
]

def init_h_votes_file():
    if not os.path.exists(H_VOTES_FILE):
        pd.DataFrame(SEED_H_VOTES, columns=["name","h1","h2","h3"]).to_csv(H_VOTES_FILE, index=False)

init_h_votes_file()

def set_bg(gif_path):
    if not os.path.exists(gif_path):
        return
    with open(gif_path,"rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    st.markdown(f"""<style>
[data-testid="stAppViewContainer"]{{background-image:url("data:image/gif;base64,{b64}");background-size:cover;background-repeat:no-repeat;background-attachment:fixed;}}
.block-container{{background-color:rgba(0,0,0,.68);border-radius:20px;padding:2rem;margin-top:2rem;}}
h1,h2,h3,h4,h5,h6{{color:#00FFFF;}}
.stButton>button{{background:linear-gradient(90deg,#4B0082,#00008B);color:white;font-weight:bold;border-radius:10px;transition:all .3s ease;}}
.stButton>button:hover{{transform:scale(1.05);box-shadow:0 0 6px #8A2BE2,0 0 22px #1E90FF;}}
</style>""", unsafe_allow_html=True)

set_bg("images/starfield.gif")

def load_scores():
    scores = {o:0 for o in OBJECTS}
    counts = {o:{"c1":0,"c2":0,"c3":0} for o in OBJECTS}
    if not os.path.exists(VOTES_FILE):
        return scores, counts
    df = pd.read_csv(VOTES_FILE)
    for _, row in df.iterrows():
        for col,pts,key in [("choice1",3,"c1"),("choice2",2,"c2"),("choice3",1,"c3")]:
            obj = str(row.get(col,"")).strip()
            if obj in scores:
                scores[obj]+=pts; counts[obj][key]+=1
    return scores, counts

def goodfor_heuristic(obj,key,counts,scores):
    c=counts[obj]; total=c["c1"]+c["c2"]+c["c3"]
    if key=="E1": return total==1 and c["c3"]==1
    if key=="E2": return total==1 and c["c2"]==1
    if key=="E3": return total==1 and c["c1"]==1
    if key=="E4": return total==2 and c["c3"]==2
    if key=="E5": return total==2 and c["c3"]==1 and c["c2"]==1 and c["c1"]==0
    if key=="E6": return scores[obj]<=3
    if key=="E7": return c["c1"]==0
    return False

def load_h_votes():
    if not os.path.exists(H_VOTES_FILE):
        return pd.DataFrame(columns=["name","h1","h2","h3"])
    return pd.read_csv(H_VOTES_FILE)

def ranked_heuristics_from_votes(df_h):
    h_scores={k:0 for k in HEURISTICS}
    for _,row in df_h.iterrows():
        for col,pts in [("h1",3),("h2",2),("h3",1)]:
            k=str(row.get(col,"")).strip()
            if k in h_scores: h_scores[k]+=pts
    return sorted(h_scores.items(),key=lambda x:-x[1])

def apply_heuristicsStep(objects_list,heuristics_order,counts,scores):
    current=list(objects_list); log=[]
    for h_key in heuristics_order:
        if len(current)<=10: break
        removed=[o for o in current if goodfor_heuristic(o,h_key,counts,scores)]
        if removed: current=[o for o in current if o not in removed]
        log.append({"Евристика":h_key,"Опис":HEURISTICS[h_key],
                    "Видалено":", ".join(removed) if removed else "—","Залишилось":len(current)})
    return current, log

def generate_expert_perms(objects_subset, n_experts=20, seed=42):
    rng = random.Random(seed)
    return [rng.sample(objects_subset, len(objects_subset)) for _ in range(n_experts)]

def firstdist(perm_a: list, perm_b: list) -> int:
    dist = 0
    for i in range(len(perm_a)):
        if perm_a[i] != perm_b[i]:
            dist += 1
    return dist

def genetic_rank(objects_subset, expert_perms, fitness_mode="sum", pop_size=1000, generations=200, mut_rate=0.15):
    """Генетичний алгоритм з великою популяцією"""
    n = len(objects_subset)
    if n == 0: return [], 0, [], [], 0

    def fitness(perm):
        dists = [firstdist(perm, exp) for exp in expert_perms]
        return -sum(dists) if fitness_mode == "sum" else -max(dists)

    def crossover(p1, p2):
        a, b = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[a:b+1] = p1[a:b+1]
        fill = [x for x in p2 if x not in child]
        j = 0
        for i in range(n):
            if child[i] is None: child[i] = fill[j]; j += 1
        return child

    def mutate(perm):
        p = perm[:]
        for i in range(n):
            if random.random() < mut_rate:
                j = random.randint(0, n - 1); p[i], p[j] = p[j], p[i]
        return p

    popul = [random.sample(objects_subset, n) for _ in range(pop_size)]
    best_perm = None; best_fit = float("-inf"); history = []; improve_iters = []; best_solutions = []

    for gen in range(generations):
        ranked_pop = sorted(popul, key=fitness, reverse=True)
        top_fit = fitness(ranked_pop[0])
        if top_fit > best_fit:
            best_fit = top_fit; best_perm = ranked_pop[0][:]; improve_iters.append(gen + 1); best_solutions = [best_perm[:]]
        elif top_fit == best_fit:
            c = ranked_pop[0][:]
            if c not in best_solutions: best_solutions.append(c)
        history.append(-best_fit)
        survivors = ranked_pop[:pop_size//2]; new_pop = survivors[:]
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(survivors, 2); new_pop.append(mutate(crossover(p1, p2)))
        popul = new_pop
    return best_perm, -best_fit, history, improve_iters, len(best_solutions)
def load_expert_triples_from_votes(votes_file, objects_subset):
    if not os.path.exists(votes_file):
        return []
    df = pd.read_csv(votes_file)
    subset_set = set(objects_subset)
    triples = []
    for _, row in df.iterrows():
        name = str(row.get("name","")).strip()
        o1 = str(row.get("choice1","")).strip()
        o2 = str(row.get("choice2","")).strip()
        o3 = str(row.get("choice3","")).strip()
        choices = [o for o in [o1,o2,o3] if o in subset_set]
        if len(choices) >= 2:
            while len(choices) < 3:
                choices.append(choices[-1])
            triples.append((name, choices[0], choices[1], choices[2]))
    return triples

def build_preference_matrix(triples, objects_subset):
    idx={o:i for i,o in enumerate(objects_subset)}
    M=[[0]*len(objects_subset) for _ in range(len(objects_subset))]
    for _,o1,o2,o3 in triples:
        for winner,losers in [(o1,[o2,o3]),(o2,[o3])]:
            if winner in idx:
                for loser in losers:
                    if loser in idx: M[idx[winner]][idx[loser]]+=1
    return pd.DataFrame(M,index=objects_subset,columns=objects_subset)

def build_rank_matrix(triples, objects_subset):
    rows=[{"Експерт":e,"1-й":o1,"2-й":o2,"3-й":o3} for e,o1,o2,o3 in triples]
    return pd.DataFrame(rows)

def cook_distance_e1(ranks_vec, triple):
    _,o1,o2,o3=triple
    pos={o:i for i,o in enumerate(ranks_vec)}
    chosen=[o for o in [o1,o2,o3] if o in pos]
    if not chosen: return 0
    sorted_chosen=sorted(chosen,key=lambda o:pos[o])
    rel_rank={o:i+1 for i,o in enumerate(sorted_chosen)}
    d=0
    for expert_rank,obj in enumerate([o1,o2,o3],start=1):
        if obj in rel_rank: d+=abs(expert_rank-rel_rank[obj])
    return d

def cook_distance_e2(ranks_vec, triple):
    _,o1,o2,o3=triple
    pos={o:i+1 for i,o in enumerate(ranks_vec)}
    d=0
    for ideal_rank,obj in enumerate([o1,o2,o3],start=1):
        if obj in pos: d+=abs(ideal_rank-pos[obj])
    return d

def brute_force_median(objects_subset, triples, heuristic="E1"):
    dist_fn = cook_distance_e1 if heuristic=="E1" else cook_distance_e2
    min_sum=float("inf"); min_max=float("inf")
    best_perms_sum=[]; best_perms_max=[]; sample_rows=[]
    for idx,perm in enumerate(itertools.permutations(objects_subset)):
        perm=list(perm)
        dists=[dist_fn(perm,t) for t in triples]
        s=sum(dists); m=max(dists)
        if idx<8:
            row={"Перестановка":" > ".join(perm)}
            for i,t in enumerate(triples): row[f"d{i+1}"]=dists[i]
            row["Сума"]=s; row["Макс"]=m; sample_rows.append(row)
        if s<min_sum: min_sum=s; best_perms_sum=[perm]
        elif s==min_sum: best_perms_sum.append(perm)
        if m<min_max: min_max=m; best_perms_max=[perm]
        elif m==min_max: best_perms_max.append(perm)
    return best_perms_sum,best_perms_max,min_sum,min_max,sample_rows

def restore_ranking(perms, objects_subset):
    rows=[{o:i+1 for i,o in enumerate(perm)} for perm in perms]
    return pd.DataFrame(rows,columns=objects_subset)

def ga_rank_cook(objects_subset,triples,heuristic="E1",fitness_mode="sum",pop_size=80,generations=300,mut_rate=0.12):
    n=len(objects_subset); dist_fn=cook_distance_e1 if heuristic=="E1" else cook_distance_e2
    def fitness(perm):
        dists=[dist_fn(perm,t) for t in triples]
        return -sum(dists) if fitness_mode=="sum" else -max(dists)
    pop=[random.sample(objects_subset,n) for _ in range(pop_size)]; best_perm=None; best_fit=float("-inf"); history=[]; improve_iters=[]
    for gen in range(generations):
        ranked_pop=sorted(pop,key=fitness,reverse=True); tf=fitness(ranked_pop[0])
        if tf>best_fit: best_fit=tf; best_perm=ranked_pop[0][:]; improve_iters.append(gen+1)
        history.append(-best_fit)
        survivors=ranked_pop[:pop_size//2]; new_pop=survivors[:]
        while len(new_pop)<pop_size:
            p1,p2=random.sample(survivors,2); new_pop.append(random.sample(objects_subset,n)) # спрощений ГА для порівняння
        pop=new_pop
    return best_perm,-best_fit,history,improve_iters

def ga_for_scale(n_objs,n_experts,fitness_mode="sum",seed=1):
    rng=random.Random(seed)
    objs=[f"O{i+1}" for i in range(n_objs)]
    triples=[(f"E{i+1}",*rng.sample(objs,3)) for i in range(n_experts)]
    return ga_rank_cook(objs,triples,heuristic="E1",fitness_mode=fitness_mode,pop_size=60,generations=200,mut_rate=0.12)

scores, counts = load_scores()

tab = st.sidebar.selectbox("Розділ",[
    "Результати ЛР1","Голосування за евристики","Застосування евристик","ЛР3","Адмін",
])

if tab=="Результати ЛР1":
    st.title("Результати лабораторної роботи №1")
    rows=[]
    for o in OBJECTS:
        rows.append({"Об'єкт":o,"1-е місце":counts[o]["c1"],"2-е місце":counts[o]["c2"],
                     "3-є місце":counts[o]["c3"],
                     "Загалом обрано раз":counts[o]["c1"]+counts[o]["c2"]+counts[o]["c3"],
                     "Сума балів":scores[o]})
    df_res=pd.DataFrame(rows).sort_values("Сума балів",ascending=False).reset_index(drop=True)
    df_res.index+=1; st.dataframe(df_res,use_container_width=True)
    fig,ax=plt.subplots(figsize=(6.5,2.5)); fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.bar(df_res["Об'єкт"].tolist(),df_res["Сума балів"].tolist(),color="white")
    ax.set_xlabel("Об'єкт",color="white"); ax.set_ylabel("Сума балів",color="white")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors="white",axis="both",labelrotation=45)
    for sp in ax.spines.values(): sp.set_color("white")
    c1,c2,c3=st.columns([1,2,1])
    with c2: st.pyplot(fig)

elif tab=="Голосування за евристики":
    st.title("Голосування за пріоритетність евристик")
    st.subheader("Перелік евристик")
    for k,v in HEURISTICS.items(): st.markdown(f"**{k}** — {v}")
    st.divider()
    name=st.text_input("Ваше ім'я")
    h_keys=list(HEURISTICS.keys())
    h1=st.selectbox("1-й пріоритет",h_keys,key="h1")
    h2=st.selectbox("2-й пріоритет",h_keys,key="h2")
    h3=st.selectbox("3-й пріоритет",h_keys,key="h3")
    if st.button("Проголосувати"):
        if not name.strip(): st.error("Введіть ім'я")
        elif len({h1,h2,h3})<3: st.error("Оберіть 3 різні евристики")
        else:
            df_h=load_h_votes()
            df_h=pd.concat([df_h,pd.DataFrame([[name.strip(),h1,h2,h3]],columns=["name","h1","h2","h3"])],ignore_index=True)
            df_h.to_csv(H_VOTES_FILE,index=False)
            st.success(f"Голос збережено. Ваш вибір: **{h1}** > **{h2}** > **{h3}**")


elif tab == "Генетичний алгоритм":
    st.title("Генетичний алгоритм (Оновлений)")
    df_h = load_h_votes();
    ranked = ranked_heuristics_from_votes(df_h)
    f_set, _ = apply_heuristicsStep(OBJECTS, [k for k, _ in ranked], counts, scores)
    f_set = sorted(f_set, key=lambda x: scores[x], reverse=True)[:10]
    expert_perms = generate_expert_perms(f_set, n_experts=20, seed=42)

    st.info(f"Об'єкти: {', '.join(f_set)}")
    if st.button("Запустити ГА"):
        p1, v1, h1, i1, n1 = genetic_rank(f_set, expert_perms, fitness_mode="sum", pop_size=1000)
        p2, v2, h2, i2, n2 = genetic_rank(f_set, expert_perms, fitness_mode="max", pop_size=1000)
        st.subheader("К1 (Сума)")
        st.write(f"Значення: {v1}, Розв'язків: {n1}");
        st.write(" > ".join(p1))
        st.subheader("К2 (Макс)")
        st.write(f"Значення: {v2}, Розв'язків: {n2}");
        st.write(" > ".join(p2))


elif tab=="Застосування евристик":
    st.title("Застосування евристик")
    df_h=load_h_votes()
    if len(df_h)==0:
        st.warning("Ще немає голосів за евристики."); st.stop()
    ranked=ranked_heuristics_from_votes(df_h)
    st.subheader("Ранжування евристик")
    st.dataframe(pd.DataFrame([{"Евристика":k,"Опис":HEURISTICS[k],"Бали":v} for k,v in ranked]),use_container_width=True)
    fig,ax=plt.subplots(figsize=(4.5,2)); fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.bar([r[0] for r in ranked],[r[1] for r in ranked],color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("white")
    ax.set_xlabel("Евристика",color="white"); ax.set_ylabel("Бали",color="white")
    c1,c2,c3=st.columns([1,2,1])
    with c2: st.pyplot(fig)
    st.divider(); st.subheader("Покрокове застосування евристик")
    ordered_keys=[k for k,_ in ranked]
    final_set,step_log=apply_heuristicsStep(OBJECTS,ordered_keys,counts,scores)
    final_set=sorted(final_set,key=lambda x:scores[x],reverse=True)[:10]
    st.dataframe(pd.DataFrame(step_log),use_container_width=True)
    st.subheader("Фінальна підмножина")
    final_df=pd.DataFrame([{"Об'єкт":o,"1-е місце":counts[o]["c1"],"2-е місце":counts[o]["c2"],
                             "3-є місце":counts[o]["c3"],"Сума балів":scores[o]} for o in final_set])
    final_df=final_df.sort_values("Сума балів",ascending=False).reset_index(drop=True); final_df.index+=1
    st.dataframe(final_df,use_container_width=True)
    if len(final_set)<=10: st.success(f"Підмножину звужено до **{len(final_set)} об'єктів**")

elif tab == "ЛР3":
    df_h = load_h_votes()
    if len(df_h)==0:
        ordered_keys=list(HEURISTICS.keys());
        ranked_h=[(k,0) for k in HEURISTICS]
    else:
        ranked_h=ranked_heuristics_from_votes(df_h);
        ordered_keys=[k for k,_ in ranked_h]

    winners_full,_=apply_heuristicsStep(OBJECTS,ordered_keys,counts,scores)
    winners=sorted(winners_full,key=lambda x:scores[x],reverse=True)[:10]
    n_winners=len(winners)

    st.header("Множинні порівняння")
    triples=load_expert_triples_from_votes(VOTES_FILE,winners)
    if not triples:
        st.warning("Не знайдено жодної трійки з votes.csv."); st.stop()
    triples_df=build_rank_matrix(triples,winners)
    st.dataframe(triples_df,use_container_width=True,hide_index=True)


    st.header("Матриця відношень переваги (1.2)")
    pref_matrix=build_preference_matrix(triples,winners)
    st.dataframe(pref_matrix,use_container_width=True)

    st.header("8. Матриця рангів за множинними порівняннями (п.1.3)")
    st.markdown("Ранг 1/2/3 = місце у МП; 0 = об'єкт не обирався цим експертом.")
    rank_mat=pd.DataFrame(0,index=[f"Ексн.{i+1}" for i in range(len(triples))],columns=winners)
    for i,(_,o1,o2,o3) in enumerate(triples):
        for rank,obj in enumerate([o1,o2,o3],start=1):
            if obj in rank_mat.columns: rank_mat.at[f"Ексн.{i+1}",obj]=rank
    st.dataframe(rank_mat,use_container_width=True)

    st.divider()

    st.header("Метрики відстані Кука")
    col_e1,col_e2=st.columns(2)
    with col_e1:
        st.markdown("""**E1 — помірна взаємність (ВИПРАВЛЕНО)**

`d = |rel(o1)-1| + |rel(o2)-2| + |rel(o3)-3|`

**Відносні** ранги серед {o1,o2,o3} у кандидаті.
Якщо o1 перший з трьох — відстань 0 для нього,
навіть якщо він на 8-му місці загалом.
Лояльний критерій.""")
    with col_e2:
        st.markdown("""**E2 — максимальне задоволення**

`d = |1-rank(o1)| + |2-rank(o2)| + |3-rank(o3)|`

**Абсолютні** ранги серед усіх n об'єктів.
o1 має бути на 1-му місці, o2 на 2-му, o3 на 3-му.
Суворіший критерій, дає більші значення.""")

    st.divider()

    st.header("9. Прямий перебір — визначення медіани Кемені")
    n_fact=math.factorial(n_winners)
    st.markdown(f"Кількість перестановок: **{n_winners}! = {n_fact:,}**")
    if n_winners>8:
        st.error(f"{n_winners}! = {n_fact:,} — прямий перебір може зависнути. Рекомендується ГА (розділ 10).")
    elif n_winners>7:
        st.warning(f"{n_winners}! = {n_fact:,} — може зайняти до хвилини.")

    heuristic_choice=st.radio("Евристика метрики Кука",
        ["E1 — помірна взаємність (відносні ранги)","E2 — максимальне задоволення (абсолютні ранги)"],
        horizontal=True,key="brute_heuristic")
    heuristic_key="E1" if "E1" in heuristic_choice else "E2"

    if st.button("Запустити прямий перебір",key="run_brute"):
        with st.spinner(f"Перебір {n_fact:,} перестановок..."):
            best_sum,best_max,min_sum,min_max,sample_rows=brute_force_median(winners,triples,heuristic=heuristic_key)

        st.subheader("9.1 Ілюстрація перших 8 перестановок (перевірка коректності, п.2.1)")
        st.dataframe(pd.DataFrame(sample_rows),use_container_width=True,hide_index=True)
        st.caption(f"d1..d{len(triples)} — відстані Кука до кожного МП. Сума і Макс — агрегати.")

        st.subheader("9.2 Мінімальні значення")
        cm1,cm2=st.columns(2); cm1.metric("Мін. сума відстаней",min_sum); cm2.metric("Мін. максимум відстані",min_max)

        st.subheader("9.3 Медіани за критерієм мін. суми відстаней")
        st.markdown(f"Знайдено **{len(best_sum)}** перестановок із сумою = {min_sum}:")
        for p in best_sum[:5]: st.markdown(f"  **{' > '.join(p)}**")
        if len(best_sum)>5: st.info(f"...та ще {len(best_sum)-5}.")

        st.subheader("9.4 Медіани за критерієм мін. максимуму відстані")
        st.markdown(f"Знайдено **{len(best_max)}** перестановок із макс. відстанню = {min_max}:")
        for p in best_max[:5]: st.markdown(f"  **{' > '.join(p)}**")
        if len(best_max)>5: st.info(f"...та ще {len(best_max)-5}.")

        st.subheader("9.5 Відновлення ранжувань об'єктів (п.4 умови)")
        st.markdown("**Ранги для медіан за мін. сумою:**")
        rs=restore_ranking(best_sum[:5],winners); rs.index=[f"Медіана {i+1}" for i in range(len(rs))]
        st.dataframe(rs,use_container_width=True)
        st.markdown("**Ранги для медіан за мін. максимумом:**")
        rm=restore_ranking(best_max[:5],winners); rm.index=[f"Медіана {i+1}" for i in range(len(rm))]
        st.dataframe(rm,use_container_width=True)

        output=io.StringIO()
        output.write("ЛР3\n\n")
        output.write(f"Евристика Кука: {heuristic_key}\nОб'єкти: {', '.join(winners)}\n\n")
        output.write(f"Мін. сума: {min_sum}\nМедіани (сума):\n")
        for p in best_sum: output.write("  "+" > ".join(p)+"\n")
        output.write(f"\nМін. макс.: {min_max}\nМедіани (макс.):\n")
        for p in best_max: output.write("  "+" > ".join(p)+"\n")
        output.write("\nМножинні порівняння:\n"+triples_df.to_string(index=False))
        output.write("\n\nМатриця переваги:\n"+pref_matrix.to_string())
        st.download_button("Зберегти результати у .txt",data=output.getvalue().encode("utf-8"),file_name="lab3_results.txt",mime="text/plain")

    st.divider()

    st.header("10. Еволюційний алгоритм (ті самі дані, що й прямий перебір)")
    ga_h=st.radio("Евристика Кука для ГА",["E1 — помірна взаємність","E2 — максимальне задоволення"],horizontal=True,key="ga_heuristic")
    ga_h_key="E1" if "E1" in ga_h else "E2"
    ga_mode=st.radio("Критерій",["Мінімізація суми","Мінімізація максимуму"],horizontal=True)
    ga_fm="sum" if "суми" in ga_mode else "max"

    if st.button("Запустити еволюційний алгоритм",key="run_ga_lr3"):
        with st.spinner("ГА..."):
            ga_perm,ga_val,ga_hist,ga_iters=ga_rank_cook(winners,triples,heuristic=ga_h_key,fitness_mode=ga_fm,pop_size=80,generations=300,mut_rate=0.12)
        label="сума" if ga_fm=="sum" else "максимум"
        st.markdown(f"Ранжування: **{' > '.join(ga_perm)}**")
        cg1,cg2,cg3=st.columns(3)
        cg1.metric(f"Найкраще ({label})",ga_val); cg2.metric("Покращень",len(ga_iters)); cg3.metric("Покоління",str(ga_iters))
        fig_ga,ax_ga=plt.subplots(figsize=(6.5,2.5)); fig_ga.patch.set_alpha(0); ax_ga.set_facecolor("none")
        ax_ga.plot(ga_hist,color="cyan",linewidth=1.5)
        for it in ga_iters: ax_ga.axvline(x=it-1,color="cyan",linestyle=":",alpha=0.5); ax_ga.text(it-1,ga_hist[it-1],str(it),color="cyan",fontsize=7,va="bottom")
        ax_ga.set_xlabel("Покоління",color="white"); ax_ga.set_ylabel(f"Найкращий {label}",color="white"); ax_ga.tick_params(colors="white")
        for sp in ax_ga.spines.values(): sp.set_color("white")
        cg1b,cg2b,cg3b=st.columns([1,3,1])
        with cg2b: st.pyplot(fig_ga)
        dist_fn_ga=cook_distance_e1 if ga_h_key=="E1" else cook_distance_e2
        dist_rows=[{"Експерт":t[0],"1-й":t[1],"2-й":t[2],"3-й":t[3],"Відстань Кука":dist_fn_ga(ga_perm,t)} for t in triples]
        dist_df=pd.DataFrame(dist_rows)
        st.subheader("Відстані від знайденого ранжування до кожного експерта")
        st.dataframe(dist_df,use_container_width=True,hide_index=True)
        cs,cm=st.columns(2); cs.metric("Сума відстаней",dist_df["Відстань Кука"].sum()); cm.metric("Максимум відстані",dist_df["Відстань Кука"].max())

    st.divider()
    st.header("11. Масштабування ГА: 20 / 50 / 100 альтернатив")
    if st.button("Запустити масштабоване тестування",key="run_scale"):
        scale_results=[]
        for n_objs,n_exps in [(20,10),(20,20),(20,30),(50,10),(50,20),(50,30),(100,10),(100,20),(100,30)]:
            with st.spinner(f"{n_objs} alt / {n_exps} exp..."):
                _,val_s,_,iters_s=ga_for_scale(n_objs,n_exps,fitness_mode="sum",seed=42)
            scale_results.append({"Альтернативи":n_objs,"Експерти":n_exps,"Мін. сума (Кук)":val_s,"Покращень":len(iters_s),"Перше покращення":iters_s[0] if iters_s else "—"})
        st.dataframe(pd.DataFrame(scale_results),use_container_width=True,hide_index=True)

    st.divider()

# ══ Адмін ══
elif tab=="Адмін":
    st.title("Адміністративна панель")
    password=st.text_input("Пароль",type="password")
    if password==ADMIN_PASSWORD:
        st.success("Доступ надано")
        st.subheader("Протокол голосування за евристики")
        df_h=load_h_votes()
        if len(df_h):
            st.dataframe(df_h,use_container_width=True)
            with open(H_VOTES_FILE,"rb") as fh:
                st.download_button("Завантажити протокол евристик",fh,"heuristic_votes.csv","text/csv")
        else: st.info("Голосів ще немає.")
        if st.button("Очистити голоси за евристики"):
            pd.DataFrame(columns=["name","h1","h2","h3"]).to_csv(H_VOTES_FILE,index=False)
            st.success("Видалено.")
        st.divider()
        st.subheader("Протокол голосування ЛР1 (votes.csv)")
        if os.path.exists(VOTES_FILE):
            df_v=pd.read_csv(VOTES_FILE); st.dataframe(df_v,use_container_width=True)
            with open(VOTES_FILE,"rb") as fh:
                st.download_button("Завантажити votes.csv",fh,"votes.csv","text/csv")
        else: st.info("Файл votes.csv не знайдено.")
        st.divider()
        st.subheader("Повний протокол ЛР3 (з іменами, конфіденційно)")
        sc2,cn2=load_scores()
        df_h2=load_h_votes()
        ok2=[k for k,_ in ranked_heuristics_from_votes(df_h2)] if len(df_h2) else list(HEURISTICS.keys())
        wf2,_=apply_heuristicsStep(OBJECTS,ok2,cn2,sc2)
        w2=sorted(wf2,key=lambda x:sc2[x],reverse=True)[:10]
        tr2=load_expert_triples_from_votes(VOTES_FILE,w2)
        if tr2:
            fp=pd.DataFrame([{"Експерт":t[0],"1-й вибір":t[1],"2-й вибір":t[2],"3-й вибір":t[3]} for t in tr2])
            st.dataframe(fp,use_container_width=True,hide_index=True)
            st.download_button("Завантажити протокол ЛР3",fp.to_csv(index=False).encode("utf-8"),"lab3_protocol.csv","text/csv")
        else: st.info("Даних з votes.csv ще немає.")
    elif password: st.error("Невірний пароль")