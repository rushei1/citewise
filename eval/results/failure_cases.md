# Failure cases

## Clustering (best configuration by mean ARI)

Configuration: `all-MiniLM-L6-v2 / agglomerative / k=true` - 26 of 180 papers sit in a cluster whose majority category is not their own.

Most confused category pairs (true -> category of the cluster it landed in):

- cs.IR -> cs.CL: 10
- cs.IR -> cs.DB: 4
- cs.CV -> cs.CL: 2
- cs.DB -> cs.CL: 2
- cs.RO -> cs.CV: 2
- cs.CL -> cs.CR: 1
- cs.CL -> cs.DB: 1
- cs.CL -> cs.IR: 1
- cs.CV -> cs.RO: 1
- cs.IR -> cs.CV: 1

Misplaced papers farthest from their own category's centroid:

- `2301.00844` (cs.IR -> cs.DB, distance 1.10): Understanding the main failure scenarios of subsea blowout preventers systems: An approach through Latent Semantic Analysis
- `2301.02817` (cs.IR -> cs.DB, distance 1.09): Cost-optimal Seeding Strategy During a Botanical Pandemic in Domesticated Fields
- `2301.00379` (cs.RO -> cs.DB, distance 1.02): A review of Implementation and Challenges of Unmanned Aerial Vehicles for Spraying Applications and Crop Monitoring in Indonesia
- `2301.03656` (cs.DB -> cs.CL, distance 0.97): Towards Multifaceted Human-Centered AI
- `2301.03774` (cs.IR -> cs.DB, distance 0.94): How Data Scientists Review the Scholarly Literature
- `2301.04780` (cs.IR -> cs.CL, distance 0.90): Much Ado About Gender: Current Practices and Future Recommendations for Appropriate Gender-Aware Information Access
- `2301.00618` (cs.CV -> cs.RO, distance 0.89): An Event-based Algorithm for Simultaneous 6-DOF Camera Pose Tracking and Mapping
- `2301.07183` (cs.IR -> cs.CL, distance 0.85): Tracking Brand-Associated Polarity-Bearing Topics in User Reviews
- `2301.00866` (cs.RO -> cs.CV, distance 0.85): 3DSGrasp: 3D Shape-Completion for Robotic Grasp
- `2301.00503` (cs.CL -> cs.IR, distance 0.85): A Concept Knowledge Graph for User Next Intent Prediction at Alipay

## Lowest-grounding LLM summaries

Papers with the smallest share of claims whose quote was found in the abstract:

- `2301.00330`: 2 verified, 2 dropped (50% verified): Efficient On-device Training via Gradient Filtering
- `2301.00399`: 2 verified, 1 dropped (67% verified): Semantic Operator Prediction and Applications
- `2301.01227`: 2 verified, 1 dropped (67% verified): Semantic Units: Organizing knowledge graphs into semantically meaningful units of representation
- `2301.01228`: 2 verified, 1 dropped (67% verified): DMOps: Data Management Operation and Recipes
- `2301.00379`: 5 verified, 2 dropped (71% verified): A review of Implementation and Challenges of Unmanned Aerial Vehicles for Spraying Applications and Crop Monitoring in Indonesia
- `2301.00823`: 6 verified, 2 dropped (75% verified): Bringing data minimization to digital wallets at scale with general-purpose zero-knowledge proofs
- `2301.00320`: 3 verified, 1 dropped (75% verified): Relevance Classification of Flood-related Twitter Posts via Multiple Transformers
- `2301.00366`: 3 verified, 1 dropped (75% verified): SS-CPGAN: Self-Supervised Cut-and-Pasting Generative Adversarial Network for Object Segmentation
- `2301.00374`: 3 verified, 1 dropped (75% verified): Optimizing Readability Using Genetic Algorithms
- `2301.00521`: 3 verified, 1 dropped (75% verified): A Policy Optimization Method Towards Optimal-time Stability

Papers with no usable output (invalid JSON after retries, or LLM error):

- `2301.00397`: Inflected Forms Are Redundant in Question Generation Models (no valid JSON after 3 attempts: Invalid \escape: line 5 column 256 (char 443))
