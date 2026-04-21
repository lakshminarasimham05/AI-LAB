"""
================================================================================
  MA3206: Artificial Intelligence  —  Assignment 4
  Assignment Scheduler: Greedy Strategies + A* Search
  ─────────────────────────────────────────────────
  Roll No. : 2301MC61
  Institute : IIT Patna  |  B.Tech Mathematics & Computing  |  Batch 2023-27
================================================================================
USAGE
  python scheduler.py                      # auto-runs all testcase*.txt files
  python scheduler.py tc1.txt tc2.txt      # run specific files
  python scheduler.py tc1.txt --no-plots   # skip plot generation (faster)

MATHEMATICAL NOTE
  Total food cost = sum of cost(food(Ai)) over ALL assignments, regardless of
  grouping. Since every student eats individually, no grouping changes the sum.
  Strategies therefore compete on NUMBER OF DAYS (schedule length).
  A* confirms the lower bound and finds any optimal grouping.
================================================================================
"""
import heapq, sys, os, time
from collections import defaultdict, deque, Counter
from itertools    import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ══════════════════════════════════════════════════════
# PALETTE
# ══════════════════════════════════════════════════════
FOOD_COLOR = {'TC':'#E63946','PM':'#2A9D8F','DF':'#E9C46A',
              'GJ':'#6A4C93','BS':'#F4A261','RG':'#457B9D'}
FOOD_FULL  = {'TC':'Tandoori Chicken','PM':'Paneer Butter Masala',
              'DF':'Dal Fry','GJ':'Gulab Jamun',
              'BS':'Biryani Special','RG':'Rajma Gravy'}
STRAT_COLOR= {'Greedy-Cost':'#E63946','Greedy-Depth':'#2A9D8F',
              'Greedy-Freq':'#E9C46A','Greedy-Topo':'#6A4C93',
              'Greedy-Hybrid':'#F4A261','A*':'#264653'}
STRAT_FULL = {'cost':'Greedy by Food Cost','depth':'Greedy by Dependency Depth',
              'frequency':'Greedy by Food-Type Frequency',
              'topological':'Greedy by Topological Order',
              'hybrid':'Greedy Hybrid (Critical-Path Cost)'}
STRAT_SHORT= {'cost':'Greedy-Cost','depth':'Greedy-Depth',
              'frequency':'Greedy-Freq','topological':'Greedy-Topo',
              'hybrid':'Greedy-Hybrid'}

plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.25,
    'grid.linestyle':'--','figure.facecolor':'white','savefig.facecolor':'white'})

def fc(food): return FOOD_COLOR.get(food,'#AAAAAA')
def shadow(ax,x,y,t,**kw):
    ax.text(x,y,t,path_effects=[pe.withStroke(linewidth=2.5,foreground='black')],**kw)

# ══════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════
def parse_input(filename):
    costs,group_size,inputs,outputs,assignments = {},1,[],[],{}
    with open(filename) as fh: raw=fh.readlines()
    for lineno,raw_line in enumerate(raw,1):
        line=raw_line.strip()
        if not line or line.startswith('%'): continue
        p=line.split(); tag=p[0].upper()
        try:
            if   tag=='C': costs[p[1]]=int(p[2])
            elif tag=='G':
                group_size=int(p[1])
                if group_size<1: raise ValueError("g>=1 required")
            elif tag=='I': inputs=[int(x) for x in p[1:] if x!='-1']
            elif tag=='O': outputs=[int(x) for x in p[1:] if x!='-1']
            elif tag=='A':
                aid=int(p[1])
                assignments[aid]={'prereqs':[int(p[2]),int(p[3])],
                                  'outcome':int(p[4]),'food':p[5]}
        except (IndexError,ValueError) as e:
            raise ValueError(f"Line {lineno}: '{line}' -> {e}")
    if not assignments: raise ValueError(f"{filename}: no assignments.")
    if not costs:       raise ValueError(f"{filename}: no food costs.")
    return {'costs':costs,'group_size':group_size,'inputs':set(inputs),
            'outputs':set(outputs),'assignments':assignments,'filename':filename}

# ══════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════
def validate(problem):
    a=problem['assignments']; warnings=[]
    produced={d['outcome'] for d in a.values()}
    avail=problem['inputs']|produced
    for aid,data in a.items():
        for p in data['prereqs']:
            if p not in avail:
                warnings.append(f"A{aid}: prereq node {p} unreachable")
    dep=build_dep(a); ind={x:len(ps) for x,ps in dep.items()}
    rev=defaultdict(set)
    for x,ps in dep.items():
        for p in ps: rev[p].add(x)
    q=deque(x for x,d in ind.items() if d==0); topo=[]; tmp=dict(ind)
    while q:
        n=q.popleft(); topo.append(n)
        for c in rev[n]:
            tmp[c]-=1
            if tmp[c]==0: q.append(c)
    if len(topo)!=len(a): warnings.append("CYCLE DETECTED!")
    return warnings

def check_schedule(schedule,problem):
    a=problem['assignments']; g=problem['group_size']
    dep=build_dep(a); done=set()
    for i,day in enumerate(schedule,1):
        assert 1<=len(day)<=g, f"Day {i}: size {len(day)} violates g={g}"
        for aid in day:
            unmet=dep[aid]-done
            assert not unmet, f"Day {i}: A{aid} prereqs {unmet} unmet"
        done.update(day)
    assert done==set(a), f"Missing: {set(a)-done}"

# ══════════════════════════════════════════════════════
# GRAPH UTILITIES
# ══════════════════════════════════════════════════════
def build_dep(assignments):
    om={d['outcome']:aid for aid,d in assignments.items()}
    return {aid:frozenset(om[p] for p in d['prereqs'] if p in om)
            for aid,d in assignments.items()}

def topo_sort(assignments,dep):
    ind={a:len(p) for a,p in dep.items()}
    rev=defaultdict(set)
    for a,ps in dep.items():
        for p in ps: rev[p].add(a)
    q=deque(sorted(a for a,d in ind.items() if d==0)); result=[]
    tmp=dict(ind)
    while q:
        n=q.popleft(); result.append(n)
        for c in sorted(rev[n]):
            tmp[c]-=1
            if tmp[c]==0: q.append(c)
    return result

def compute_depths(assignments,dep):
    d={a:0 for a in assignments}; ch=True
    while ch:
        ch=False
        for a in assignments:
            for p in dep[a]:
                if d[p]+1>d[a]: d[a]=d[p]+1; ch=True
    return d

def compute_downstream(assignments,dep):
    rev=defaultdict(set)
    for a,ps in dep.items():
        for p in ps: rev[p].add(a)
    r={}
    for s in assignments:
        vis,stk=set(),[s]
        while stk:
            n=stk.pop()
            for c in rev[n]:
                if c not in vis: vis.add(c); stk.append(c)
        r[s]=len(vis)
    return r

def compute_cp_cost(assignments,dep,costs):
    """Critical-path cost: most expensive path from this node to any leaf."""
    rev=defaultdict(set)
    for a,ps in dep.items():
        for p in ps: rev[p].add(a)
    order=topo_sort(assignments,dep); cp={}
    for a in reversed(order):
        own=costs[assignments[a]['food']]
        ch=[cp[c] for c in rev[a] if c in cp]
        cp[a]=own+(max(ch) if ch else 0)
    return cp

def get_available(remaining,done,dep):
    return sorted(a for a in remaining if dep[a].issubset(done))

# ══════════════════════════════════════════════════════
# COST HELPERS
# ══════════════════════════════════════════════════════
def day_cost(day,assignments,costs):
    return sum(costs[assignments[a]['food']] for a in day)
def menu_str(day,assignments):
    c=Counter(assignments[a]['food'] for a in day)
    return ', '.join(f"{v}-{k}" for k,v in sorted(c.items()))
def total_cost(schedule,assignments,costs):
    return sum(day_cost(d,assignments,costs) for d in schedule)

# ══════════════════════════════════════════════════════
# GREEDY  (5 strategies)
# ══════════════════════════════════════════════════════
def greedy_schedule(problem,strategy='cost'):
    """
    Generic greedy scheduler supporting five strategies.

    At each day:
      1. Identify available assignments (all prerequisites satisfied).
      2. Sort by the chosen criterion.
      3. Pick the top min(|available|, g) assignments.
      4. Mark done; repeat.

    Strategies
    ----------
    cost        Ascending food cost. Minimises each day's bill locally.
                Pro: direct local cost minimisation.
                Con: may delay critical-path nodes, increasing total days.

    depth       Descending downstream-assignment count (critical-path first).
                Pro: unlocks the most future work each day => fewest total days.
                Con: ignores food costs (all schedules have identical total cost).

    frequency   Descending frequency of the food type among remaining tasks.
                Pro: clusters same-food assignments => homogeneous daily menus.
                Con: frequency signal decays; similar to cost on small instances.

    topological Ascending topological depth (BFS-level order).
                Pro: earliest possible scheduling; predictable and fair.
                Con: no awareness of cost or downstream impact.

    hybrid      Descending critical-path cost weight, then ascending food cost.
                The cp-cost = sum of food costs along the most expensive path
                from this assignment to any leaf.
                Pro: balances urgency with local cost — a middle ground between
                'depth' (max throughput) and 'cost' (min daily spend).
    """
    a=problem['assignments']; costs=problem['costs']; g=problem['group_size']
    dep=build_dep(a)
    depths=compute_depths(a,dep)
    downstream=compute_downstream(a,dep)
    cp=compute_cp_cost(a,dep,costs)

    remaining,done,schedule=set(a),set(),[]
    while remaining:
        av=get_available(remaining,done,dep)
        if not av: break
        if strategy=='cost':
            av.sort(key=lambda x:(costs[a[x]['food']],x))
        elif strategy=='depth':
            av.sort(key=lambda x:(-downstream[x],-depths[x],x))
        elif strategy=='frequency':
            fr=Counter(a[x]['food'] for x in remaining)
            av.sort(key=lambda x:(-fr[a[x]['food']],costs[a[x]['food']],x))
        elif strategy=='topological':
            av.sort(key=lambda x:(depths[x],x))
        elif strategy=='hybrid':
            av.sort(key=lambda x:(-cp.get(x,0),costs[a[x]['food']],x))
        day=av[:g]; schedule.append(day)
        done.update(day); remaining-=set(day)
    return schedule

# ══════════════════════════════════════════════════════
# A* SEARCH
# ══════════════════════════════════════════════════════
def astar_schedule(problem,time_limit=120.0):
    """
    A* search for minimum total-food-cost schedule.

    STATE        frozenset of completed assignment IDs.
    g(n)         total food cost accumulated from start to n.
    h(n)         Σ cost(food(Ai)) for all Ai not yet done.

    ADMISSIBILITY PROOF
      (1) Every remaining assignment must be solved (no skipping).
      (2) Each student individually consumes their food item (no sharing).
      (3) Therefore true remaining cost h*(n) >= h(n).  Hence h <= h*. [admissible]
      (4) In fact h(n) = h*(n): every food cost is unavoidable regardless of
          grouping or ordering.  h is an EXACT heuristic.

    CONSISTENCY (MONOTONICITY) PROOF
      For transition n -> n' by solving day-set D:
        c(n,D,n') = Σ_{Ai in D} cost(food(Ai))
        h(n) - h(n') = Σ_{Ai in D} cost(food(Ai)) = c(n,D,n')
      So h(n) <= c(n,D,n') + h(n') holds with equality. [consistent]
      => A* with this heuristic expands each state at most once (optimal + complete).

    BRANCHING
      All subsets of available assignments of size 1..min(|available|,g).
      Integer counter breaks heap ties (avoids comparing frozensets).
    """
    a=problem['assignments']; costs=problem['costs']; g=problem['group_size']
    dep=build_dep(a); all_ids=frozenset(a)
    fc_map={x:costs[a[x]['food']] for x in a}
    total=sum(fc_map.values())
    def h(done): return total-sum(fc_map[x] for x in done)

    counter=0
    heap=[(h(frozenset()),0,0,frozenset(),[])]
    best_g={}; explored=0; t0=time.time()

    while heap:
        if time.time()-t0>time_limit:
            print(f"    [A*] Time limit reached after {explored} states."); break
        f_val,g_cost,_,done,sched=heapq.heappop(heap)
        if done in best_g and best_g[done]<=g_cost: continue
        best_g[done]=g_cost; explored+=1
        if done==all_ids: return sched,g_cost,explored
        av=get_available(all_ids-done,done,dep)
        if not av: continue
        pick=min(len(av),g)
        for size in range(pick,0,-1):
            for combo in combinations(av,size):
                nd=done|frozenset(combo)
                ng=g_cost+sum(fc_map[x] for x in combo)
                if nd not in best_g or best_g[nd]>ng:
                    counter+=1
                    heapq.heappush(heap,(ng+h(nd),ng,counter,nd,sched+[list(combo)]))
    return None,float('inf'),explored

# ══════════════════════════════════════════════════════
# PRINT HELPERS
# ══════════════════════════════════════════════════════
B='\033[1m'; R='\033[0m'; G='\033[32m'; Y='\033[33m'; C='\033[36m'

def print_schedule(schedule,problem,label):
    a=problem['assignments']; costs=problem['costs']
    print(f"\n  {B}Strategy: {label}{R}")
    total=0
    for i,d in enumerate(schedule,1):
        dc=day_cost(d,a,costs); ms=menu_str(d,a)
        astr=', '.join(f"A{x}" for x in sorted(d))
        total+=dc
        print(f"    Day-{i:>2}: {astr:<38} Menu: {ms:<22} Cost: {dc}")
    print(f"    {'-'*75}")
    print(f"    {B}Total Days: {len(schedule):>3}   Total Cost: {total}{R}")
    return total

def print_astar(schedule,cost,explored,elapsed,problem,greedy_res):
    a=problem['assignments']; costs=problem['costs']
    print(f"\n  {'-'*66}")
    print(f"  {B}{C}A* SEARCH  -  GLOBALLY OPTIMAL RESULT{R}")
    print(f"  {'-'*66}")
    for i,d in enumerate(schedule,1):
        dc=day_cost(d,a,costs); ms=menu_str(d,a)
        astr=', '.join(f"A{x}" for x in sorted(d))
        print(f"    Day-{i:>2}: {astr:<38} Menu: {ms:<22} Cost: {dc}")
    tc=total_cost(schedule,a,costs)
    print(f"    {'-'*75}")
    print(f"    {B}Total Days: {len(schedule)}   Total Cost: {tc}   States: {explored}   Time: {elapsed:.4f}s{R}")
    best=min(greedy_res,key=lambda k:(greedy_res[k]['cost'],greedy_res[k]['days']))
    bc=greedy_res[best]['cost']; bd=greedy_res[best]['days']
    print(f"\n    {'Metric':<36} {'A*':>8} {'Best Greedy':>12}")
    print(f"    {'-'*58}")
    print(f"    {'Total Food Cost':<36} {cost:>8} {bc:>12}")
    print(f"    {'Total Days':<36} {len(schedule):>8} {bd:>12}")
    print(f"    {'Cost Improvement (Greedy - A*)':<36} {bc-cost:>8} {'---':>12}")
    print(f"    {'Day Difference  (A* - Greedy)':<36} {len(schedule)-bd:>8} {'---':>12}")
    print(f"    {'States Explored by A*':<36} {explored:>8} {'N/A':>12}")
    print(f"\n    {Y}[Best Greedy Strategy: {best}]{R}")

# ══════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════
def plot_dag(problem,save_path):
    a=problem['assignments']; dep=build_dep(a)
    depths=compute_depths(a,dep); downstream=compute_downstream(a,dep)
    layers=defaultdict(list)
    for x,d in depths.items(): layers[d].append(x)
    for d in layers: layers[d].sort()
    max_d=max(layers); max_w=max(len(v) for v in layers.values())

    fig,ax=plt.subplots(figsize=(max(11,(max_d+1)*2.8),max(5,max_w*2.0+2.2)))
    ax.set_facecolor('#F7F9FC'); fig.patch.set_facecolor('#F7F9FC'); ax.axis('off')
    ax.set_xlim(-0.6,max_d+0.6); ax.set_ylim(-0.9,max_w+0.5)

    pos={}
    for d,aids in layers.items():
        n=len(aids)
        for i,x in enumerate(aids): pos[x]=(d,(max_w-n)/2.0+i)

    # Layer shading
    for d in range(max_d+1):
        shade='#EEF2F7' if d%2==0 else '#E4EBF5'
        ax.axvspan(d-0.45,d+0.45,color=shade,zorder=0,alpha=0.55)
        ax.text(d,max_w+0.22,f'Layer {d}',ha='center',va='bottom',
                fontsize=7.5,color='#777',style='italic')

    rev=defaultdict(set)
    for x,ps in dep.items():
        for p in ps: rev[p].add(x)

    for src,children in rev.items():
        x1,y1=pos[src]
        for dst in children:
            x2,y2=pos[dst]
            rad=0.12 if abs(y1-y2)<0.1 else 0.05
            ax.annotate('',xy=(x2-0.24,y2),xytext=(x1+0.24,y1),
                arrowprops=dict(arrowstyle='->',color='#445566',lw=1.6,
                                connectionstyle=f'arc3,rad={rad}'),zorder=1)

    R=0.22
    for x,(px,py) in pos.items():
        food=a[x]['food']; color=fc(food); cost=problem['costs'].get(food,'?')
        ds=downstream[x]
        if ds>=3:
            glow=plt.Circle((px,py),R+0.07,color=color,alpha=0.22,zorder=2)
            ax.add_patch(glow)
        circ=plt.Circle((px,py),R,facecolor=color,edgecolor='#1a1a2e',lw=1.8,zorder=3)
        ax.add_patch(circ)
        shadow(ax,px,py+0.01,f'A{x}',ha='center',va='center',
               fontsize=8.5,fontweight='bold',color='white',zorder=4)
        ax.text(px,py-R-0.14,f'{food} (Rs.{cost})',ha='center',va='top',
                fontsize=6.8,color='#333',zorder=4)

    seen=sorted({a[x]['food'] for x in a})
    patches=[mpatches.Patch(color=fc(f),label=f"{f} - {FOOD_FULL.get(f,f)} (Rs.{problem['costs'].get(f,'?')})")
             for f in seen]
    ax.legend(handles=patches,loc='lower right',fontsize=8.5,framealpha=0.95,
              title='Food Items',title_fontsize=8,edgecolor='#ccc')
    ax.set_title('Assignment Dependency DAG\n'
                 'Colour = food required  |  Glow = high downstream impact',
                 fontsize=11,fontweight='bold',pad=14,color='#1a1a2e')
    plt.tight_layout()
    plt.savefig(save_path,dpi=160,bbox_inches='tight'); plt.close()


def plot_gantt(schedule,problem,title,save_path):
    a=problem['assignments']; costs=problem['costs']
    g=problem['group_size']; n=len(schedule)
    fig_w=max(10,g*2.2+4.5); fig_h=max(3.8,n*0.75+1.8)
    fig,(ax,ax_c)=plt.subplots(1,2,figsize=(fig_w+3.5,fig_h),
                                gridspec_kw={'width_ratios':[fig_w,2.8]})
    used=set(); cum=0; cum_pts=[]
    for di,day in enumerate(schedule):
        y=n-di-1; dc=day_cost(day,a,costs); cum+=dc; cum_pts.append((di+1,cum))
        ax.axhspan(y-0.38,y+0.38,color='#F0F4F8' if di%2==0 else 'white',
                   zorder=0,alpha=0.7)
        for slot,aid in enumerate(sorted(day)):
            food=a[aid]['food']; color=fc(food); used.add(food)
            rect=plt.Rectangle((slot,y-0.30),0.88,0.60,
                                facecolor=color,edgecolor='#1a1a2e',lw=0.9,zorder=2)
            ax.add_patch(rect)
            shadow(ax,slot+0.44,y,f'A{aid}',ha='center',va='center',
                   fontsize=8.5,fontweight='bold',color='white',zorder=3)
        ax.text(g+0.12,y,f'Rs.{dc}',va='center',fontsize=8.5,color='#333',fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'Day {i+1}' for i in range(n)][::-1],fontsize=9)
    ax.set_xlim(-0.08,g+0.9); ax.set_ylim(-0.55,n-0.45)
    ax.set_xlabel('Assignment slot within day',fontsize=9)
    ax.set_title(title,fontsize=10.5,fontweight='bold',pad=9,color='#1a1a2e')
    tc=total_cost(schedule,a,costs)
    ax.text(0.99,1.01,f'Total Cost: Rs.{tc}  |  Days: {n}',
            transform=ax.transAxes,ha='right',va='bottom',fontsize=9,color='#333',fontweight='bold')
    patches=[mpatches.Patch(color=fc(f),label=f"{f} {FOOD_FULL.get(f,f)}")
             for f in sorted(used)]
    ax.legend(handles=patches,loc='lower right',fontsize=7.5,framealpha=0.92)
    # Cumulative
    xs=[0]+[c for _,c in cum_pts]; ys=list(range(len(xs)))
    ax_c.step(xs,ys,where='post',color='#264653',lw=2)
    ax_c.fill_betweenx(ys,0,xs,alpha=0.18,color='#264653',step='post')
    ax_c.set_yticks(range(n+1))
    ax_c.set_yticklabels(['Start']+[f'D{i+1}' for i in range(n)],fontsize=8)
    ax_c.set_xlabel('Cumulative cost',fontsize=8)
    ax_c.set_title('Cumulative\nCost',fontsize=9,fontweight='bold'); ax_c.invert_yaxis()
    plt.tight_layout(w_pad=0.5)
    plt.savefig(save_path,dpi=160,bbox_inches='tight'); plt.close()


def plot_dashboard(all_results,basename,save_path):
    strats=list(all_results.keys())
    cv=[all_results[s]['cost'] for s in strats]
    dv=[all_results[s]['days'] for s in strats]
    ev=[c/d if d else 0 for c,d in zip(cv,dv)]
    colors=[STRAT_COLOR.get(s,'#888') for s in strats]

    fig=plt.figure(figsize=(18,10))
    gs=fig.add_gridspec(2,2,hspace=0.45,wspace=0.38)
    axes=[fig.add_subplot(gs[i//2,i%2]) for i in range(3)]
    fig.suptitle(f'Strategy Comparison Dashboard  -  {basename}',
                 fontsize=14,fontweight='bold',color='#1a1a2e',y=0.99)

    def bar(ax,vals,ylabel,title):
        bars=ax.bar(range(len(strats)),vals,color=colors,ec='#1a1a2e',lw=0.8,width=0.55,zorder=2)
        ax.set_xticks(range(len(strats))); ax.set_xticklabels(strats,rotation=28,ha='right',fontsize=8.5)
        ax.set_ylabel(ylabel,fontsize=9); ax.set_title(title,fontsize=10.5,fontweight='bold',color='#1a1a2e')
        mv=min(vals)
        for b,v in zip(bars,vals):
            lbl=f'{v:.2f}' if isinstance(v,float) else str(v)
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(vals)*0.012,lbl,
                    ha='center',va='bottom',fontsize=10,fontweight='bold',color='#1a1a2e')
            if v==mv: b.set_edgecolor('#FFD700'); b.set_linewidth(3.2); b.set_zorder(3)
        ax.set_ylim(0,max(vals)*1.2); ax.grid(axis='y',alpha=0.3)

    bar(axes[0],cv,'Total Food Cost (Rs.)','Total Cost  (lower = better  [gold border])')
    bar(axes[1],dv,'Total Days','Total Days  (lower = better  [gold border])')
    bar(axes[2],ev,'Cost per Day (Rs./day)','Cost Efficiency  (cost / days)')

    # Radar chart
    ax_r=fig.add_subplot(gs[1,1],polar=True)
    cats=['Cost','Days','Efficiency']; n_c=len(cats)
    angs=[k*2*np.pi/n_c for k in range(n_c)]+[0]
    def norm_inv(vals):
        mn,mx=min(vals),max(vals)
        if mx==mn: return [1.0]*len(vals)
        return [1-(v-mn)/(mx-mn) for v in vals]
    nc=norm_inv(cv); nd=norm_inv(dv); ne=norm_inv(ev)
    ax_r.set_theta_offset(np.pi/2); ax_r.set_theta_direction(-1)
    ax_r.set_xticks([k*2*np.pi/n_c for k in range(n_c)])
    ax_r.set_xticklabels(cats,fontsize=10)
    ax_r.set_ylim(0,1.05); ax_r.set_yticks([0.25,0.5,0.75,1.0])
    ax_r.set_yticklabels(['25%','50%','75%','100%'],fontsize=7)
    for i,(name,vals) in enumerate(zip(strats,zip(nc,nd,ne))):
        v=list(vals)+[vals[0]]
        ax_r.plot(angs,v,color=colors[i],lw=2.2,label=name)
        ax_r.fill(angs,v,color=colors[i],alpha=0.10)
    ax_r.set_title('Normalised Performance\n(larger area = better)',
                   fontsize=10,fontweight='bold',pad=14,color='#1a1a2e')
    ax_r.legend(loc='upper right',bbox_to_anchor=(1.50,1.18),fontsize=8,framealpha=0.9)
    plt.savefig(save_path,dpi=160,bbox_inches='tight'); plt.close()


def plot_curves(all_schedules,problem,save_path):
    a=problem['assignments']; costs=problem['costs']
    max_days=max(len(s) for s in all_schedules.values())
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8),sharex=False)
    fig.suptitle('Per-Day Cost & Cumulative Cost per Strategy',
                 fontsize=12,fontweight='bold',color='#1a1a2e')
    strats=list(all_schedules.keys()); w=0.13
    for i,(name,sched) in enumerate(all_schedules.items()):
        dc=[day_cost(d,a,costs) for d in sched]
        col=STRAT_COLOR.get(name,'#888')
        off=(i-len(strats)/2+0.5)*w
        ax1.bar(np.arange(1,len(dc)+1)+off,dc,width=w,color=col,label=name,
                ec='#1a1a2e',lw=0.5,alpha=0.9)
        cum=list(np.cumsum(dc))
        ax2.plot(range(1,len(cum)+1),cum,marker='o',ms=5,label=name,
                 color=col,lw=2.2,ls='--')
        ax2.fill_between(range(1,len(cum)+1),0,cum,alpha=0.08,color=col)
    ax1.set_ylabel('Daily Cost (Rs.)'); ax1.set_xlabel('Day')
    ax1.set_title('Per-Day Cost Breakdown',fontsize=10.5,fontweight='bold')
    ax1.legend(fontsize=8,loc='upper right')
    ax1.set_xticks(range(1,max_days+1))
    ax2.set_ylabel('Cumulative Cost (Rs.)'); ax2.set_xlabel('Day')
    ax2.set_title('Cumulative Food Cost',fontsize=10.5,fontweight='bold')
    ax2.legend(fontsize=8,loc='lower right')
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(save_path,dpi=160,bbox_inches='tight'); plt.close()


def plot_heatmap(all_schedules,problem,save_path):
    a=problem['assignments']; costs=problem['costs']
    foods=sorted({d['food'] for d in a.values()}); strats=list(all_schedules.keys())
    cnt_mat=[]; cost_mat=[]
    for name,sched in all_schedules.items():
        cnt=Counter(a[x]['food'] for d in sched for x in d)
        cnt_mat.append([cnt.get(f,0) for f in foods])
        cost_mat.append([cnt.get(f,0)*costs.get(f,0) for f in foods])
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(max(10,len(foods)*2.4+3),max(4,len(strats)+2)))
    fig.suptitle('Food Usage Heatmap per Strategy',fontsize=12,fontweight='bold',color='#1a1a2e')
    def hm(ax,data,title):
        data=np.array(data,dtype=float)
        im=ax.imshow(data,cmap='YlOrRd',aspect='auto',vmin=0)
        plt.colorbar(im,ax=ax,shrink=0.8)
        ax.set_xticks(range(len(foods)))
        ax.set_xticklabels([f"{f}\n{FOOD_FULL.get(f,f)}\n(Rs.{costs.get(f,'?')})"
                            for f in foods],fontsize=8.5)
        ax.set_yticks(range(len(strats))); ax.set_yticklabels(strats,fontsize=9)
        ax.set_title(title,fontsize=10.5,fontweight='bold')
        for i in range(len(strats)):
            for j in range(len(foods)):
                v=data[i,j]
                ax.text(j,i,int(v),ha='center',va='center',fontsize=11,fontweight='bold',
                        color='white' if v>data.max()*0.55 else '#1a1a2e')
    hm(ax1,cnt_mat,'Servings Count'); hm(ax2,cost_mat,'Cost Contribution (Rs.)')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(save_path,dpi=160,bbox_inches='tight'); plt.close()


def plot_astar_vs_greedy(astar_sched,best_sched,best_name,problem,save_path):
    a=problem['assignments']; costs=problem['costs']; g=problem['group_size']
    def draw(ax,schedule,label,tc):
        n=len(schedule); used=set()
        for di,day in enumerate(schedule):
            y=n-di-1; dc=day_cost(day,a,costs)
            ax.axhspan(y-0.38,y+0.38,color='#F0F4F8' if di%2==0 else 'white',
                       zorder=0,alpha=0.6)
            for slot,aid in enumerate(sorted(day)):
                food=a[aid]['food']; color=fc(food); used.add(food)
                rect=plt.Rectangle((slot,y-0.30),0.88,0.60,
                                   facecolor=color,edgecolor='#1a1a2e',lw=0.9,zorder=2)
                ax.add_patch(rect)
                shadow(ax,slot+0.44,y,f'A{aid}',ha='center',va='center',
                       fontsize=8.5,fontweight='bold',color='white',zorder=3)
            ax.text(g+0.12,y,f'Rs.{dc}',va='center',fontsize=8.5,color='#333')
        ax.set_yticks(range(n))
        ax.set_yticklabels([f'Day {i+1}' for i in range(n)][::-1],fontsize=8.5)
        ax.set_xlim(-0.08,g+0.9); ax.set_ylim(-0.55,n-0.45)
        ax.set_title(f'{label}\nDays: {n}   Total Cost: Rs.{tc}',
                     fontsize=10.5,fontweight='bold',color='#1a1a2e')
        patches=[mpatches.Patch(color=fc(f),label=f) for f in sorted(used)]
        ax.legend(handles=patches,fontsize=7.5,loc='lower right')
    n1,n2=len(astar_sched),len(best_sched)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,max(5,max(n1,n2)*0.8+2.2)))
    draw(ax1,astar_sched,'A* Search - Globally Optimal Cost',
         total_cost(astar_sched,a,costs))
    draw(ax2,best_sched,f'Best Greedy ({best_name})',
         total_cost(best_sched,a,costs))
    fig.suptitle('A* vs Best Greedy - Side-by-Side Schedule Comparison',
                 fontsize=12,fontweight='bold',color='#1a1a2e',y=1.01)
    plt.tight_layout()
    plt.savefig(save_path,dpi=160,bbox_inches='tight'); plt.close()


def plot_summary_table(all_results,basename,save_path):
    strats=list(all_results.keys())
    col_labels=['Strategy','Total Cost','Total Days','Cost/Day','Min Days?','Optimal Cost?']
    min_cost=min(v['cost'] for v in all_results.values())
    min_days=min(v['days'] for v in all_results.values())
    rows=[]
    for s in strats:
        c=all_results[s]['cost']; d=all_results[s]['days']
        rows.append([s,str(c),str(d),f"{c/d:.2f}" if d else "--",
                     'YES' if d==min_days else 'no',
                     'YES' if c==min_cost else 'no'])
    fig,ax=plt.subplots(figsize=(13,0.6*len(strats)+1.6))
    ax.axis('off')
    cc=[]
    for row in rows:
        r=['#F7F9FC']*6
        if row[4]=='YES': r[2]='#C8E6FF'
        if row[5]=='YES': r[1]='#C8F7C5'
        cc.append(r)
    tbl=ax.table(cellText=rows,colLabels=col_labels,cellLoc='center',
                 loc='center',cellColours=cc)
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2,1.8)
    for (r,c),cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor('#1a1a2e'); cell.set_text_props(color='white',fontweight='bold')
        cell.set_edgecolor('#CCCCCC')
    ax.set_title(f'Results Summary - {basename}',fontsize=11,fontweight='bold',pad=8,color='#1a1a2e')
    plt.tight_layout()
    plt.savefig(save_path,dpi=160,bbox_inches='tight'); plt.close()

# ══════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════
def run_one(input_file,output_dir='output',make_plots=True):
    os.makedirs(output_dir,exist_ok=True)
    base=os.path.splitext(os.path.basename(input_file))[0]
    print(f"\n{'='*66}")
    print(f"  FILE: {input_file}")
    problem=parse_input(input_file)
    warnings=validate(problem)
    for w in warnings: print(f"  [WARN] {w}")
    a=problem['assignments']; costs=problem['costs']; g=problem['group_size']
    n=len(a); total_food=sum(costs[d['food']] for d in a.values())
    lb=-(-n//g)
    print(f"  Assignments: {n}  |  Group size g={g}")
    print(f"  Food costs : {costs}")
    print(f"  Fixed total food cost (any valid schedule): Rs.{total_food}")
    print(f"  Lower bound on days (ceil(n/g))           : {lb} days")
    print(f"{'='*66}")

    all_strats=['cost','depth','frequency','topological','hybrid']
    greedy_res={}; greedy_scheds={}
    for key in all_strats:
        sched=greedy_schedule(problem,strategy=key)
        cost=print_schedule(sched,problem,STRAT_FULL[key])
        short=STRAT_SHORT[key]
        try: check_schedule(sched,problem)
        except AssertionError as e: print(f"  [SCHEDULE ERROR] {e}")
        greedy_res[short]={'cost':cost,'days':len(sched)}
        greedy_scheds[short]=sched
        if make_plots:
            plot_gantt(sched,problem,f'Schedule: {STRAT_FULL[key]}',
                       os.path.join(output_dir,f'{base}_{key}_gantt.png'))

    best=min(greedy_res,key=lambda k:(greedy_res[k]['cost'],greedy_res[k]['days']))
    bc,bd=greedy_res[best]['cost'],greedy_res[best]['days']
    print(f"\n  {G}[Best Greedy: {best}  Cost=Rs.{bc}  Days={bd}]{R}")

    print(f"\n  Running A* Search...")
    t0=time.time()
    astar_sched,astar_cost,explored=astar_schedule(problem)
    elapsed=time.time()-t0
    print(f"  Completed: {elapsed:.4f}s | States explored: {explored}")

    all_res=dict(greedy_res); all_scheds=dict(greedy_scheds)
    if astar_sched:
        try: check_schedule(astar_sched,problem)
        except AssertionError as e: print(f"  [A* ERROR] {e}")
        print_astar(astar_sched,astar_cost,explored,elapsed,problem,greedy_res)
        all_res['A*']={'cost':astar_cost,'days':len(astar_sched),
                       'states':explored,'time_s':elapsed}
        all_scheds['A*']=astar_sched
        if make_plots:
            plot_gantt(astar_sched,problem,'Schedule: A* Search (Globally Optimal)',
                       os.path.join(output_dir,f'{base}_astar_gantt.png'))
            plot_astar_vs_greedy(astar_sched,greedy_scheds[best],best,problem,
                                 os.path.join(output_dir,f'{base}_astar_vs_greedy.png'))

    if make_plots:
        plot_dag(problem,os.path.join(output_dir,f'{base}_dag.png'))
        plot_dashboard(all_res,base,os.path.join(output_dir,f'{base}_dashboard.png'))
        plot_curves(all_scheds,problem,os.path.join(output_dir,f'{base}_curves.png'))
        plot_heatmap(all_scheds,problem,os.path.join(output_dir,f'{base}_heatmap.png'))
        plot_summary_table(all_res,base,os.path.join(output_dir,f'{base}_summary_table.png'))

    return greedy_res,all_res,all_scheds

if __name__=='__main__':
    args=sys.argv[1:]; no_plots='--no-plots' in args
    files=[a for a in args if not a.startswith('--')]
    if not files:
        files=sorted(f for f in os.listdir('.') if f.startswith('testcase') and f.endswith('.txt'))
        if not files: print("Usage: python scheduler.py [file.txt ...] [--no-plots]"); sys.exit(1)
    for fname in files:
        if not os.path.isfile(fname): print(f"[ERROR] Not found: {fname}"); continue
        try: run_one(fname,output_dir='output',make_plots=not no_plots)
        except Exception as e:
            print(f"\n[ERROR] {fname}: {e}")
            import traceback; traceback.print_exc()
