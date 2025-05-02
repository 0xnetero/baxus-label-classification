# OCR Failure Analysis Report

## Summary Statistics

Total failed cases: 268
Average confidence score: 0.83
Median confidence score: 0.85

## Error Pattern Distribution

| Pattern           |   Count | Percentage   |
|:------------------|--------:|:-------------|
| no_prediction     |       9 | 3.4%         |
| missing_key_words |      45 | 16.8%        |
| brand_confusion   |     136 | 50.7%        |
| minor_variation   |      27 | 10.1%        |
| other             |      51 | 19.0%        |

## Most Commonly Missing Tokens

| Token    |   Missing Count |
|:---------|----------------:|
| year     |              33 |
| batch    |              29 |
| bourbon  |              23 |
| barrel   |              17 |
| release  |              11 |
| proof    |               9 |
| edition  |               8 |
| stagg    |               8 |
| michters |               8 |
| 10       |               6 |
| rye      |               6 |
| strength |               6 |
| eagle    |               5 |
| rare     |               5 |
| 7        |               5 |
| straight |               5 |
| limited  |               5 |
| weller   |               5 |
| single   |               5 |
| 2024     |               5 |

## Example Cases by Error Pattern

### No Prediction

Total cases: 9

**Example 1:** 1046
- Ground Truth: Double Eagle Very Rare
- Predicted: None
- OCR Tokens: 
- Confidence: 0.00

**Example 2:** 11663
- Ground Truth: Clase Azul Reposado
- Predicted: None
- OCR Tokens: 
- Confidence: 0.00

**Example 3:** 15946
- Ground Truth: Smoke Wagon Uncut Unfiltered Bourbon
- Predicted: None
- OCR Tokens: 
- Confidence: 0.00

**Example 4:** 15947
- Ground Truth: Smoke Wagon Small Batch Bourbon
- Predicted: None
- OCR Tokens: 
- Confidence: 0.00

**Example 5:** 2028
- Ground Truth: Jack Daniel's Single Barrel Eric Church
- Predicted: None
- OCR Tokens: 
- Confidence: 0.00


### Missing Key Words

Total cases: 45

**Example 1:** 1066
- Ground Truth: Jack Daniel's No. 27 Gold Maple Wood Finish
- Predicted: Jack Daniel's
- OCR Tokens: JACK DANIELS, GOL
- Confidence: 0.92

**Example 2:** 1142
- Ground Truth: Buffalo Trace Kosher Rye 7 Year
- Predicted: Buffalo Trace
- OCR Tokens: BUFFALOTRACE, PASTALLERY, KOSHER, SERAIGNZRYE, KENTUCKY, WHISXEY, STRAIGHT RYE
- Confidence: 0.92

**Example 3:** 12843
- Ground Truth: Blanton's Straight from the Barrel
- Predicted: Blanton's
- OCR Tokens: Blanten's, CHTFAOMTHE, ROURBON, 9-19-17, PROOF, 130-3, H, BLANTOST
- Confidence: 0.78

**Example 4:** 13266
- Ground Truth: Heaven Hill Bottled In Bond 7 Year
- Predicted: Heaven Hill Grain to Glass Kentucky Straight Bourbon Whiskey
- OCR Tokens: VEN HILL OISTILL, BOTTLED-IN-BOND, HEAVEN HILL, 1004, 50%, ALC/VO, BOURBON WHISKEY, KENTUCKY STRAIGHT
- Confidence: 0.86

**Example 5:** 13581
- Ground Truth: Four Roses 2020 Limited Edition Small Batch OESV
- Predicted: Four Roses Bourbon
- OCR Tokens: 2020, FourRoses.
- Confidence: 0.64


### Brand Confusion

Total cases: 136

**Example 1:** 1074
- Ground Truth: Early Times Bottled In Bond
- Predicted: Michter's US1 Kentucky Straight Bourbon
- OCR Tokens: 100, EARLYS, TIMES, BOURBON WHISKY, KENTUCKY STRAIGHT, BOTTLED-IN-BOND
- Confidence: 0.78

**Example 2:** 1211
- Ground Truth: Old Rip Van Winkle 10 Year
- Predicted: Tito's Handmade Vodka
- OCR Tokens: ORIP, HANDMADE HOUN
- Confidence: 0.67

**Example 3:** 12112
- Ground Truth: Fortaleza Blanco Tequila
- Predicted: Blanton's
- OCR Tokens: LEA, PRODUCT OF MEXICO, HECHO EN MEXI, ESTATE, DEAG, STONE, BOTILED, IDOE
- Confidence: 0.56

**Example 4:** 1217
- Ground Truth: Joseph Magnus Cigar Blend Bourbon
- Predicted: Bulleit Bourbon
- OCR Tokens: ABL, EMA, ROURBO, CIGAR BLEND, HISKEY, &COGISAC CASKS, SHERRY
- Confidence: 0.60

**Example 5:** 1282
- Ground Truth: Willett Family Estate Small Batch Rye 4 Year
- Predicted: Angel's Envy Cask Strength
- OCR Tokens: Hans Bottle at Cask Strength, Willett., Millett
- Confidence: 0.69


### Minor Variation

Total cases: 27

**Example 1:** 1165
- Ground Truth: Four Roses Single Barrel Straight Bourbon
- Predicted: Four Roses Single Barrel
- OCR Tokens: FourRoses., SINGLE BARREL, ocky Straight Bourbon, SN46-3A
- Confidence: 0.96

**Example 2:** 13668
- Ground Truth: Maker's Mark 101
- Predicted: Maker's Mark 46
- OCR Tokens: Maker's, Mark, Maker's, WHISKY, Mark, BATCHED FORA, HIGHER PROOF, 101, PRROE, WHISKY, Maker's, SMark, 101, WHISKY, 750ml ale.50.5Zvol
- Confidence: 0.80

**Example 3:** 149
- Ground Truth: Woodford Reserve Double Oaked
- Predicted: Woodford Reserve Bourbon
- OCR Tokens: IOUBLE OAKEI, BARRELFINISH SELECT, WOODFORD RESERVE, DOUBLEOAKED, DOUBLE, OAKED, 45.2%ALC/VOL [30.4PRO0F, KENTUCKY STRAIGHT BOURBON WHISKEYE, FINISHED INA SECONO OAK BARREL-
- Confidence: 1.00

**Example 4:** 1579
- Ground Truth: Basil Hayden's 10 Year
- Predicted: Basil Hayden's 10 Year Bourbon
- OCR Tokens: ARTFULLYAGED, BASIL, HAYDEN'S, Kentucky Straight, Bourbon Whiskey, rafted using the HIGH RYE, BOU R BON recipe that makes, Basil Hayden's Bourbon so uniquc., this special rekase is aged I0 YEARS,, adding COMPLEXITY and deeper, flavor to an already intriguing spirit., 10, AGED, YEARS, DIST LLED AND BOTTLED BY, KENTUCKY SPRINGS DISTILLING CO., CLERMONT-FRANKFORT,KENTUCKY USA, 750ML 40%ALC./VOL.(80PR00F)
- Confidence: 1.00

**Example 5:** 15966
- Ground Truth: Knob Creek 9 Year
- Predicted: Knob Creek Rye
- OCR Tokens: CLERMONT, KENTUCKY, KNOB, HESIO, COMPAI, CREEK, 6, KENTUCKY STRAIGHT BOURBON, AGEI, NINE, WHISKEY, YEAI, KNOB, CREEK, SMALLBATCH, 100PR00F, SINCE, 50%ALC/VOL, 100PROOF, 1992
- Confidence: 0.93


### Other

Total cases: 51

**Example 1:** 13089
- Ground Truth: Elijah Craig Toasted Barrel
- Predicted: Elijah Craig Small Batch
- OCR Tokens: TOASTED BARREL, ELIJAH, CRAIG., FastcdBarrl, 1789, KENTUCKY STRAIGHT, BOURBON WHISKEY, TOASTED BARREL, OUR SMALL BATCH BOURBON FINISHED, in TOASTED NEW OAK BARRELS, 94, FATHER ofBOURBON, PROOF:, 47%ALC/VOL, 1789
- Confidence: 1.00

**Example 2:** 14303
- Ground Truth: Elijah Craig Barrel Proof Batch C917
- Predicted: Elijah Craig Small Batch
- OCR Tokens: BARREL PROI, ELIJAH, CRAIG, SmallBatch, KENTUCKY STRAIGHT, BOURBON WHISKEY, BARREL PROOF, UNCUT.STRAIGHT FROMTHE BARREL, 131.0, 65.5%, C917, PROOF, ALC/VOL, BATCHN
- Confidence: 1.00

**Example 3:** 144
- Ground Truth: Elijah Craig Barrel Proof Batch A119
- Predicted: Elijah Craig Small Batch
- OCR Tokens: SARRELPROOI, ELIJAH, CRAIG., Small Batch, 1789, KENTUCKY STRAIGHT, BOURBON WHISKEY, BARREL PROOF, 135.2, 67.6%, A119, PROOF, ALC/VOL
- Confidence: 1.00

**Example 4:** 158
- Ground Truth: Weller Antique 107
- Predicted: Weller 12 Year The Original Wheated Bourbon
- OCR Tokens: WHEATED BOURBON, THE ORIGINAL, ANTIQUE 107
- Confidence: 0.82

**Example 5:** 15984
- Ground Truth: Old Forester 150th Anniversary
- Predicted: Old Forester 86 Proof
- OCR Tokens: FORESTER, 1870152020, BATCH PROOF, OLD, UNFILTERED, FORESTER, KENTUCKY STRAIGHT, BOURBON WHISKY, 15
- Confidence: 1.00


## Recommendations for Improvement

### Brand Confusion
- Increase weight for brand name matching in the fuzzy matching algorithm
- Add specific brand name detection pre-processing

### Missing Key Words
- Improve OCR model to better detect small or stylized text
- Consider using image preprocessing techniques to enhance text visibility

### Minor Variations
- Fine-tune the string normalization process
- Add more synonym mappings for common terms (e.g., "whisky" vs "whiskey")

