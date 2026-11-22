# CLI search examples (Task 4)

```
Query: What is a deployer under the AI Act?
Top-5 (score = cosine similarity = 1 - Chroma distance)

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
Query: Which uses of AI are completely banned in the EU?
Top-5 (score = cosine similarity = 1 - Chroma distance)

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
Query: Do I have to tell users that they are chatting with an AI and not a human?
Top-5 (score = cosine similarity = 1 - Chroma distance)

1. chunk_id=intro_177  score=0.5354  pages=33-33  binding=False
   (132) Certain AI systems intended to interact with natural persons or to generate content may pose specific risks of
2. chunk_id=article_50_part_1  score=0.5031  pages=82-82  binding=True
   TRANSPARENCY OBLIGATIONS FOR PROVIDERS AND DEPLOYERS OF CERTAIN AI SYSTEMS
3. chunk_id=intro_165  score=0.4734  pages=31-31  binding=False
   (119) Considering the quick pace of innovation and the technological evolution of digital services in scope of different
4. chunk_id=intro_96  score=0.4690  pages=18-18  binding=False
   (63) The fact that an AI system is classified as a high-risk AI system under this Regulation should not be interpreted a
5. chunk_id=intro_28  score=0.4533  pages=6-6  binding=False
   benefits, risks, safeguards, rights and obligations in relation to the use of AI systems. In cooperation with the releva
```

```
Query: Must a human be able to supervise a high-risk AI system?
Top-5 (score = cosine similarity = 1 - Chroma distance)

1. chunk_id=article_14_part_1  score=0.7405  pages=60-60  binding=True
   1. High-risk AI systems shall be designed and developed in such a way, including with appropriate human-machine
2. chunk_id=article_14_part_2  score=0.6880  pages=60-61  binding=True
   Article 14 – Human oversight
3. chunk_id=article_26_part_5  score=0.6435  pages=69-69  binding=True
   Article 26 – Obligations of deployers of high-risk AI systems
4. chunk_id=intro_109  score=0.6203  pages=21-21  binding=False
   on the market or put it into service. High-risk AI systems should be designed in a manner to enable deployers to
5. chunk_id=intro_131  score=0.6159  pages=24-25  binding=False
   (91) Given the nature of AI systems and the risks to safety and fundamental rights possibly associated with their use,
```

```
Query: From which date do the bans on prohibited AI practices apply?
Top-5 (score = cosine similarity = 1 - Chroma distance)

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

## Same ban query with --binding-only

```
Query: Which uses of AI are completely banned in the EU?
Top-5 (score = cosine similarity = 1 - Chroma distance)

1. chunk_id=article_109  score=0.6767  pages=121-121  binding=True
   In Article 11 of Regulation (EU) 2019/2144, the following paragraph is added:
2. chunk_id=article_108  score=0.6737  pages=120-121  binding=True
   Regulation (EU) 2018/1139 is amended as follows:
3. chunk_id=article_104  score=0.6631  pages=119-119  binding=True
   In Article 22(5) of Regulation (EU) No 168/2013, the following subparagraph is added:
4. chunk_id=article_103  score=0.6602  pages=118-118  binding=True
   In Article 17(5) of Regulation (EU) No 167/2013, the following subparagraph is added:
5. chunk_id=article_102  score=0.6543  pages=118-118  binding=True
   CHAPTER XIII
```
