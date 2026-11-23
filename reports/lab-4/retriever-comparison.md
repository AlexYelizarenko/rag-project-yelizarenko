# Retriever comparison: dense vs BM25 vs hybrid (Lab 4, Task 2–3)

Generated with `uv run python -m retrieval.search "<query>" --mode <mode> --k 5`.

## What is a deployer under the AI Act?

```
Query: What is a deployer under the AI Act?
Mode: dense  Top-5 (score = cosine similarity = 1 - Chroma distance)

1. chunk_id=article_3_def_4  score=0.7541  pages=46-46  binding=True
   Article 3 – Definitions
2. chunk_id=intro_131  score=0.6530  pages=24-25  binding=False
   (91) Given the nature of AI systems and the risks to safety and fundamental rights possibly associated with their use,
3. chunk_id=intro_109  score=0.6335  pages=21-21  binding=False
   on the market or put it into service. High-risk AI systems should be designed in a manner to enable deployers to
4. chunk_id=article_26_part_1  score=0.6170  pages=67-68  binding=True
   Obligations of deployers of high-risk AI systems
5. chunk_id=article_26_part_5  score=0.6159  pages=69-69  binding=True
   Article 26 – Obligations of deployers of high-risk AI systems
```

```
Query: What is a deployer under the AI Act?
Mode: bm25  Top-5 (score = BM25 (Okapi) score, not comparable across queries)

1. chunk_id=article_91  score=14.7879  pages=111-112  binding=True
   Power to request documentation and information
2. chunk_id=article_3_def_61  score=14.0933  pages=50-50  binding=True
   Article 3 – Definitions
3. chunk_id=article_3_def_4  score=13.4189  pages=46-46  binding=True
   Article 3 – Definitions
4. chunk_id=article_26_part_4  score=13.1287  pages=68-69  binding=True
   Article 26 – Obligations of deployers of high-risk AI systems
5. chunk_id=article_86  score=13.1223  pages=110-110  binding=True
   Right to explanation of individual decision-making
```

```
Query: What is a deployer under the AI Act?
Mode: hybrid  Top-5 (score = RRF, sum of 1/(60 + rank) over dense and BM25 (top-50 each))

1. chunk_id=article_3_def_4  score=0.0323  pages=46-46  binding=True  dense_rank=1  bm25_rank=3
   Article 3 – Definitions
2. chunk_id=article_26_part_1  score=0.0286  pages=67-68  binding=True  dense_rank=4  bm25_rank=17
   Obligations of deployers of high-risk AI systems
3. chunk_id=annex_IV_part_1  score=0.0275  pages=130-130  binding=True  dense_rank=21  bm25_rank=6
   Technical documentation referred to in Article 11(1)
4. chunk_id=intro_131  score=0.0274  pages=24-25  binding=False  dense_rank=2  bm25_rank=29
   (91) Given the nature of AI systems and the risks to safety and fundamental rights possibly associated with their use,
5. chunk_id=article_13_part_1  score=0.0268  pages=59-59  binding=True  dense_rank=7  bm25_rank=24
   Transparency and provision of information to deployers
```

## Which uses of AI are completely banned in the EU?

```
Query: Which uses of AI are completely banned in the EU?
Mode: dense  Top-5 (score = cosine similarity = 1 - Chroma distance)

1. chunk_id=intro_80  score=0.7045  pages=15-15  binding=False
   should be obliged to register the AI system in the EU database established under this Regulation. With a view to
2. chunk_id=article_109  score=0.6767  pages=121-121  binding=True
   In Article 11 of Regulation (EU) 2019/2144, the following paragraph is added:
3. chunk_id=article_108  score=0.6737  pages=120-121  binding=True
   Regulation (EU) 2018/1139 is amended as follows:
4. chunk_id=intro_41  score=0.6719  pages=8-8  binding=False
   (28) Aside from the many beneficial uses of AI, it can also be misused and provide novel and powerful tools for
5. chunk_id=article_104  score=0.6631  pages=119-119  binding=True
   In Article 22(5) of Regulation (EU) No 168/2013, the following subparagraph is added:
```

```
Query: Which uses of AI are completely banned in the EU?
Mode: bm25  Top-5 (score = BM25 (Okapi) score, not comparable across queries)

1. chunk_id=intro_165  score=16.1434  pages=31-31  binding=False
   (119) Considering the quick pace of innovation and the technological evolution of digital services in scope of different
2. chunk_id=intro_110  score=14.8808  pages=21-21  binding=False
   and use the AI system correctly and as appropriate. In order to enhance legibility and accessibility of the information
3. chunk_id=intro_78  score=14.7949  pages=14-15  binding=False
   No 109/2011, (EU) No 458/2011, (EU) No 65/2012, (EU) No 130/2012, (EU) No 347/2012, (EU) No 351/2012, (EU)
4. chunk_id=intro_101  score=14.7407  pages=19-19  binding=False
   the provider should cover uses of AI systems which, while not directly covered by the intended purpose and
5. chunk_id=intro_41  score=14.1134  pages=8-8  binding=False
   (28) Aside from the many beneficial uses of AI, it can also be misused and provide novel and powerful tools for
```

```
Query: Which uses of AI are completely banned in the EU?
Mode: hybrid  Top-5 (score = RRF, sum of 1/(60 + rank) over dense and BM25 (top-50 each))

1. chunk_id=intro_41  score=0.0310  pages=8-8  binding=False  dense_rank=4  bm25_rank=5
   (28) Aside from the many beneficial uses of AI, it can also be misused and provide novel and powerful tools for
2. chunk_id=intro_80  score=0.0294  pages=15-15  binding=False  dense_rank=1  bm25_rank=17
   should be obliged to register the AI system in the EU database established under this Regulation. With a view to
3. chunk_id=intro_165  score=0.0287  pages=31-31  binding=False  dense_rank=21  bm25_rank=1
   (119) Considering the quick pace of innovation and the technological evolution of digital services in scope of different
4. chunk_id=intro_175  score=0.0278  pages=33-33  binding=False  dense_rank=13  bm25_rank=11
   (131) In order to facilitate the work of the Commission and the Member States in the AI field as well as to increase the
5. chunk_id=article_40_part_1  score=0.0246  pages=76-76  binding=True  dense_rank=25  bm25_rank=18
   Standards, conformity assessment, certificates, registration
```

## What does Article 50 require for chatbots?

```
Query: What does Article 50 require for chatbots?
Mode: dense  Top-5 (score = cosine similarity = 1 - Chroma distance)

1. chunk_id=intro_165  score=0.5089  pages=31-31  binding=False
   (119) Considering the quick pace of innovation and the technological evolution of digital services in scope of different
2. chunk_id=article_53_part_3  score=0.4383  pages=85-85  binding=True
   Article 53 – Obligations for providers of general-purpose AI models
3. chunk_id=article_3_def_15  score=0.4268  pages=47-47  binding=True
   Article 3 – Definitions
4. chunk_id=article_31_part_3  score=0.4237  pages=72-72  binding=True
   Article 31 – Requirements relating to notified bodies
5. chunk_id=annex_XI_part_1  score=0.4234  pages=141-141  binding=True
   Technical documentation referred to in Article 53(1), point (a) — technical documentation for
```

```
Query: What does Article 50 require for chatbots?
Mode: bm25  Top-5 (score = BM25 (Okapi) score, not comparable across queries)

1. chunk_id=intro_230  score=10.1355  pages=43-44  binding=False
   (54) Directive (EU) 2019/1937 of the European Parliament and of the Council of 23 October 2019 on the protection of pers
2. chunk_id=article_2_part_3  score=9.0452  pages=46-46  binding=True
   Article 2 – Scope
3. chunk_id=article_26_part_5  score=8.4763  pages=69-69  binding=True
   Article 26 – Obligations of deployers of high-risk AI systems
4. chunk_id=article_3_def_50  score=8.3147  pages=49-49  binding=True
   Article 3 – Definitions
5. chunk_id=article_79_part_2  score=7.5803  pages=106-107  binding=True
   Article 79 – Procedure at national level for dealing with AI systems presenting a risk
```

```
Query: What does Article 50 require for chatbots?
Mode: hybrid  Top-5 (score = RRF, sum of 1/(60 + rank) over dense and BM25 (top-50 each))

1. chunk_id=article_50_part_3  score=0.0290  pages=83-83  binding=True  dense_rank=8  bm25_rank=10
   Article 50 – Transparency obligations for providers and deployers of certain AI systems
2. chunk_id=article_91  score=0.0267  pages=111-112  binding=True  dense_rank=16  bm25_rank=14
   Power to request documentation and information
3. chunk_id=article_50_part_2  score=0.0243  pages=82-82  binding=True  dense_rank=44  bm25_rank=8
   Article 50 – Transparency obligations for providers and deployers of certain AI systems
4. chunk_id=annex_IV_part_2  score=0.0219  pages=130-131  binding=True  dense_rank=23  bm25_rank=41
   Annex IV – Technical documentation referred to in Article 11(1)
5. chunk_id=article_50_part_1  score=0.0215  pages=82-82  binding=True  dense_rank=43  bm25_rank=25
   TRANSPARENCY OBLIGATIONS FOR PROVIDERS AND DEPLOYERS OF CERTAIN AI SYSTEMS
```

## What is the maximum fine for using a prohibited AI practice?

```
Query: What is the maximum fine for using a prohibited AI practice?
Mode: dense  Top-5 (score = cosine similarity = 1 - Chroma distance)

1. chunk_id=article_5_part_1  score=0.6558  pages=51-52  binding=True
   CHAPTER II
2. chunk_id=article_101_part_1  score=0.6083  pages=117-118  binding=True
   Fines for providers of general-purpose AI models
3. chunk_id=intro_223  score=0.6033  pages=42-42  binding=False
   (169) Compliance with the obligations on providers of general-purpose AI models imposed under this Regulation should
4. chunk_id=article_60_part_1  score=0.6027  pages=92-92  binding=True
   Testing of high-risk AI systems in real world conditions outside AI regulatory sandboxes
5. chunk_id=article_101_part_2  score=0.5968  pages=118-118  binding=True
   Article 101 – Fines for providers of general-purpose AI models
```

```
Query: What is the maximum fine for using a prohibited AI practice?
Mode: bm25  Top-5 (score = BM25 (Okapi) score, not comparable across queries)

1. chunk_id=intro_163  score=20.7319  pages=30-31  binding=False
   (117) The codes of practice should represent a central tool for the proper compliance with the obligations provided for
2. chunk_id=article_101_part_1  score=17.8568  pages=117-118  binding=True
   Fines for providers of general-purpose AI models
3. chunk_id=article_5_part_1  score=15.5995  pages=51-52  binding=True
   CHAPTER II
4. chunk_id=article_56_part_3  score=15.2908  pages=87-87  binding=True
   Article 56 – Codes of practice
5. chunk_id=article_56_part_1  score=15.1626  pages=86-87  binding=True
   SECTION 4
```

```
Query: What is the maximum fine for using a prohibited AI practice?
Mode: hybrid  Top-5 (score = RRF, sum of 1/(60 + rank) over dense and BM25 (top-50 each))

1. chunk_id=article_5_part_1  score=0.0323  pages=51-52  binding=True  dense_rank=1  bm25_rank=3
   CHAPTER II
2. chunk_id=article_101_part_1  score=0.0323  pages=117-118  binding=True  dense_rank=2  bm25_rank=2
   Fines for providers of general-purpose AI models
3. chunk_id=intro_207  score=0.0273  pages=39-39  binding=False  dense_rank=10  bm25_rank=17
   (157) This Regulation is without prejudice to the competences, tasks, powers and independence of relevant national publi
4. chunk_id=article_56_part_3  score=0.0269  pages=87-87  binding=True  dense_rank=29  bm25_rank=4
   Article 56 – Codes of practice
5. chunk_id=intro_101  score=0.0244  pages=19-19  binding=False  dense_rank=18  bm25_rank=26
   the provider should cover uses of AI systems which, while not directly covered by the intended purpose and
```

## From which date do the bans on prohibited AI practices apply?

```
Query: From which date do the bans on prohibited AI practices apply?
Mode: dense  Top-5 (score = cosine similarity = 1 - Chroma distance)

1. chunk_id=intro_232  score=0.6944  pages=44-44  binding=False
   (179) This Regulation should apply from 2 August 2026. However, taking into account the unacceptable risk associated
2. chunk_id=article_109  score=0.6497  pages=121-121  binding=True
   In Article 11 of Regulation (EU) 2019/2144, the following paragraph is added:
3. chunk_id=article_108  score=0.6429  pages=120-121  binding=True
   Regulation (EU) 2018/1139 is amended as follows:
4. chunk_id=article_106  score=0.6316  pages=119-119  binding=True
   In Article 5 of Directive (EU) 2016/797, the following paragraph is added:
5. chunk_id=article_107  score=0.6307  pages=120-120  binding=True
   In Article 5 of Regulation (EU) 2018/858 the following paragraph is added:
```

```
Query: From which date do the bans on prohibited AI practices apply?
Mode: bm25  Top-5 (score = BM25 (Okapi) score, not comparable across queries)

1. chunk_id=intro_232  score=15.7658  pages=44-44  binding=False
   (179) This Regulation should apply from 2 August 2026. However, taking into account the unacceptable risk associated
2. chunk_id=intro_44  score=15.3009  pages=8-9  binding=False
   (16) Directive (EU) 2019/882 of the European Parliament and of the Council of 17 April 2019 on the accessibility require
3. chunk_id=intro_66  score=14.8293  pages=12-12  binding=False
   (45) Practices that are prohibited by Union law, including data protection law, non-discrimination law, consumer
4. chunk_id=intro_139  score=14.2439  pages=26-26  binding=False
   (97) The notion of general-purpose AI models should be clearly defined and set apart from the notion of AI systems to
5. chunk_id=intro_47  score=14.0078  pages=9-9  binding=False
   (31) AI systems providing social scoring of natural persons by public or private actors may lead to discriminatory
```

```
Query: From which date do the bans on prohibited AI practices apply?
Mode: hybrid  Top-5 (score = RRF, sum of 1/(60 + rank) over dense and BM25 (top-50 each))

1. chunk_id=intro_232  score=0.0328  pages=44-44  binding=False  dense_rank=1  bm25_rank=1
   (179) This Regulation should apply from 2 August 2026. However, taking into account the unacceptable risk associated
2. chunk_id=intro_207  score=0.0288  pages=39-39  binding=False  dense_rank=11  bm25_rank=8
   (157) This Regulation is without prejudice to the competences, tasks, powers and independence of relevant national publi
3. chunk_id=intro_44  score=0.0283  pages=8-9  binding=False  dense_rank=22  bm25_rank=2
   (16) Directive (EU) 2019/882 of the European Parliament and of the Council of 17 April 2019 on the accessibility require
4. chunk_id=article_5_part_1  score=0.0278  pages=51-52  binding=True  dense_rank=13  bm25_rank=11
   CHAPTER II
5. chunk_id=intro_41  score=0.0274  pages=8-8  binding=False  dense_rank=20  bm25_rank=7
   (28) Aside from the many beneficial uses of AI, it can also be misused and provide novel and powerful tools for
```

