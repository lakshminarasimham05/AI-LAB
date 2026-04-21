================================================================================
  MA3206: Artificial Intelligence — Assignment 4
  Assignment Scheduler: Greedy Strategies + A* Search
  Roll No. : 2301MC61  |  B.Tech Mathematics & Computing  |  IIT Patna
================================================================================

FILES IN THIS SUBMISSION
  scheduler.py    Main Python program (all algorithms + plots)
  testcase1.txt   Original Fig.1 problem  (11 assignments, g=3, uniform costs)
  testcase2.txt   Research Lab workflow   (14 assignments, g=2, skewed costs)
  testcase3.txt   Software Sprint         (16 assignments, g=4, varied costs)
  README.txt      This file
  report.pdf      Full PDF report (proofs, tables, 15+ plots, analysis)

DEPENDENCIES
  Python >= 3.8
  matplotlib    pip install matplotlib
  numpy         pip install numpy
  reportlab     pip install reportlab   (only for re-generating report.pdf)

HOW TO RUN
  # Run all test cases (auto-detected by testcase*.txt glob)
  python scheduler.py

  # Run specific files
  python scheduler.py testcase1.txt testcase2.txt testcase3.txt

  # Skip plot generation (faster, text output only)
  python scheduler.py testcase1.txt --no-plots

  # Re-generate the PDF report (requires output/ plots to exist)
  python build_report.py

  Output plots are saved to ./output/<basename>_<type>.png

INPUT FILE FORMAT
  % Comment lines (ignored)
  C  <FOOD> <COST>                     food item and cost per serving
  G  <g>                               max assignments solvable per day
  I  <id1> <id2> ... -1                input node IDs (books/notes)
  O  <id1> <id2> ... -1                output node IDs (final outcomes)
  A  <id> <pre1> <pre2> <out> <FOOD>   assignment: id, two prereq nodes,
                                       outcome node, food item required

  Rules:
  - A prereq node must be either an I-line input or the outcome of an assignment.
  - Each assignment has exactly 2 prerequisites and 1 outcome node.
  - The program validates all prereqs and detects cycles before scheduling.

GREEDY STRATEGIES (5 implemented)
  1. Greedy by Food Cost      Ascending food cost. Minimise each day's bill.
  2. Greedy by Depth          Descending downstream count. Critical-path first.
                              BEST strategy: achieves minimum days.
  3. Greedy by Frequency      Descending food-type frequency. Cluster same foods.
  4. Greedy by Topological    Ascending topological depth. BFS-level order.
  5. Greedy Hybrid [NEW]      Descending critical-path cost weight, then ascending
                              food cost. Balances urgency with cost-awareness.
                              Matches Depth on day-count; adds cost sensitivity.

A* SEARCH
  State   : frozenset of completed assignment IDs.
  g(n)    : total food cost accumulated from start to n.
  h(n)    : sum of food costs of all remaining unsolved assignments.

  ADMISSIBILITY:
    Every remaining assignment must be solved; food is individually consumed.
    True remaining cost h*(n) = h(n) exactly (exact heuristic, not just a bound).
    => h(n) <= h*(n) always: admissible. [QED]

  CONSISTENCY:
    Transition cost c(n,D,n') = sum of food costs for day-set D.
    h(n) - h(n') = sum of food costs for D = c(n,D,n').
    => h(n) <= c(n,D,n') + h(n') with equality: consistent. [QED]

  RESULT: A* is provably optimal and explores only 96-171 states in < 5ms.

MATHEMATICAL KEY INSIGHT
  Total food cost = sum of cost(food(Ai)) for ALL assignments.
  This is FIXED for any valid schedule (all students eat individually, no sharing).
  Strategies therefore compete on NUMBER OF DAYS, not on cost.
  A* confirms this by finding the same total cost as every greedy strategy.

OUTPUT FILES PER TEST CASE
  output/<base>_dag.png            Layered dependency DAG with food colours
  output/<base>_cost_gantt.png     Gantt + cumulative chart: Greedy-Cost
  output/<base>_depth_gantt.png    Gantt + cumulative chart: Greedy-Depth
  output/<base>_frequency_gantt.png  Greedy-Freq
  output/<base>_topological_gantt.png  Greedy-Topo
  output/<base>_hybrid_gantt.png   Greedy-Hybrid
  output/<base>_astar_gantt.png    A* optimal schedule
  output/<base>_astar_vs_greedy.png  Side-by-side A* vs Best Greedy
  output/<base>_dashboard.png      4-panel comparison + radar chart
  output/<base>_curves.png         Per-day bars + cumulative lines
  output/<base>_heatmap.png        Food serving + cost contribution heatmaps
  output/<base>_summary_table.png  Styled results table

EXAMPLE OUTPUT (Test Case 1)
  Strategy: Greedy by Dependency Depth
    Day- 1: A1, A2, A6   Menu: 3-TC   Cost: 3
    Day- 2: A3, A5, A10  Menu: 3-TC   Cost: 3
    Day- 3: A4, A7, A9   Menu: 1-DF, 2-PM   Cost: 3
    Day- 4: A8, A11      Menu: 1-DF, 1-GJ   Cost: 2
    Total Days: 4   Total Cost: Rs.11

  A* SEARCH - GLOBALLY OPTIMAL RESULT
    Total Days: 4   Total Cost: Rs.11   States: 171   Time: 0.004s
    Cost Improvement over Greedy: 0  (greedy already optimal on this instance)
================================================================================
