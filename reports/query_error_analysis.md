# Query-level analysis of frozen ESCI results

Descriptive post-hoc analysis of frozen official judged-candidate test rankings; not whole-catalog retrieval, no training/tuning.

| Comparison | Mean NDCG difference | Improved / worsened / unchanged queries | Exploratory95% interval |
|---|---:|---:|---:|
| structured minus plain | +0.01215 | 4029 / 3021 / 1906 | [+0.01028, +0.01401] |
| structured minus doc2query_t5 | +0.00392 | 3795 / 3486 / 1675 | [+0.00205, +0.00577] |
| structured minus ling_doc2query | -0.00200 | 3317 / 3434 / 2205 | [-0.00356, -0.00023] |
| e5 minus structured | +0.02075 | 4606 / 3897 / 453 | [+0.01690, +0.02436] |
| hybrid_structured minus e5 | -0.00128 | 4035 / 4125 / 796 | [-0.00344, +0.00100] |
| hybrid_plain minus e5 | -0.00787 | 3817 / 4325 / 814 | [-0.01012, -0.00560] |

Intervals use1,000 paired query resamples, seed42; exploratory, no multiple-comparison correction.

| Grouping | Group | Queries | Plain | Structured | Ling queries | E5 | Structured+E5 |
|---|---|---:|---:|---:|---:|---:|---:|
| source | behavioral | 223 | 0.6794 | 0.6914 | 0.6959 | 0.7082 | 0.7023 |
| source | negations | 689 | 0.5258 | 0.5331 | 0.5327 | 0.5480 | 0.5418 |
| source | nlqec | 30 | 0.6145 | 0.6442 | 0.6382 | 0.6871 | 0.6793 |
| source | other | 7233 | 0.7202 | 0.7329 | 0.7348 | 0.7546 | 0.7538 |
| source | parse_pattern | 781 | 0.6857 | 0.6971 | 0.7017 | 0.7138 | 0.7143 |
| query_length | 1 | 271 | 0.7236 | 0.7166 | 0.7146 | 0.7440 | 0.7442 |
| query_length | 2 | 1441 | 0.7063 | 0.7139 | 0.7174 | 0.7472 | 0.7403 |
| query_length | 3-5 | 5925 | 0.7045 | 0.7183 | 0.7204 | 0.7378 | 0.7368 |
| query_length | 6+ | 1319 | 0.6742 | 0.6877 | 0.6886 | 0.6991 | 0.7022 |

Dataset source names are metadata, not independently verified query types. Small strata should not support general claims.

## Examples for manual inspection

These deliberately selected extremes show possible failure modes; the JSON also includes seeded random examples. Product labels: E exact, S substitute, C complement, I irrelevant. Expansion text and all eight top-five rankings are preserved in query_error_analysis.json.

### structured minus plain: largest_losses, query107558
Query: utv flag
NDCG difference: -0.5299
- plain rank1: [E] Forever Wave USA Subdued Thin Blue Line Punisher Flag (B074BC9YD1)
- structured rank1: [C] Tusk UTV Horizontal Flag Mount 1 3/4"-1 7/8" - Fits: Arctic Cat WILDCAT 1000i H.O. 2012-2016 (B003AYUXGE)

### structured minus plain: largest_losses, query75759
Query: onses for men
NDCG difference: -0.5000
- plain rank1: [E] Disney Men's Finding Dory Uniform Union Suit, Blue, L (B01JHBCBEU)
- structured rank1: [I] Quiksilver Men's Mountain and Wave Black Hat, White, Small/Medium (B014UAX228)

### structured minus plain: largest_losses, query105964
Query: tv stand for 65 inch tv
NDCG difference: -0.4581
- plain rank1: [E] Mecor Modern LED Black TV Stand,12 Colors LED TV Stand with Remote Control Lights,High Gloss TV Cabinet with Storage&2 Drawers,65 Inch Entertainment Center for Living Room (B081342DX9)
- structured rank1: [S] Walker Edison Milton Classic 2 Shelf Corner TV Stand for TVs up to 65 Inches, 58 Inch, Driftwood (B01L17ISAY)

### structured minus plain: largest_losses, query45758
Query: giroptic giroptic
NDCG difference: -0.4532
- plain rank1: [E] Giroptic iO HD 360 degree camera for iPhone/iPad (B06Y5YTQGV)
- structured rank1: [C] CameronSino Replacement Battery Compatible with Giroptic Camera Edges Closer to 360, Edges Closer to 360° Action ca (B07ZFYNTNR)

### structured minus plain: largest_losses, query20258
Query: bubbling cauldron halloween
NDCG difference: -0.4459
- plain rank1: [I] iCandy Combat Pick Your Poison Print Black and White Skeleton Bubbling Cauldron Picture Halloween Wall Decoration Seasonal Poster (B015M9HYXO)
- structured rank1: [I] Vinyl Wall Decals Quotes Sayings Words Art Decor Lettering Vinyl Wall Art Witches Bubbling Cauldron for Halloween (B07JDXD9N8)

### structured minus plain: largest_gains, query62477
Query: lightweight makeup organizer
NDCG difference: +0.5967
- plain rank1: [S] Acrylic Makeup Organizer, 360 Degree Rotating Adjustable Cosmetic Storage Display Case with 8 Layers, Large Capacity, Fits Lipsticks, bottle, Crema and More, Clear Transparent (17.1inch Height) (B087D65JZJ)
- structured rank1: [E] Vera Bradley Cotton Brush Up Cosmetic Makeup Organizer Case, Foxwood (B07XYY99FD)

### structured minus plain: largest_gains, query96432
Query: space base board game
NDCG difference: +0.5915
- plain rank1: [S] Splendor Board Game (Base Game) | Family Board Game | Board Game for Adults and Family | Strategy Game | Ages 10+ | 2 to 4 players | Average Playtime 30 minutes | Made by Space Cowboys (B00IZEUFIA)
- structured rank1: [E] Space Base: The Emergence of Shy Pluto (B07MTMLRVY)

### structured minus plain: largest_gains, query13584
Query: bath and body works air plug in
NDCG difference: +0.5715
- plain rank1: [C] Bath and Body Works 4 Pack Linen and Lavender Wallflowers Fragrance Refill. 0.8 fl oz. (B07MQ7J7F5)
- structured rank1: [E] Bath&body Works White Wallflowers Pluggable Home Fragrance Diffuser Pack of 2 (B00W0IKK50)

### structured minus plain: largest_gains, query95512
Query: so vegan in 5 cookbook
NDCG difference: +0.5670
- plain rank1: [S] Go lean vegan, vegan cookbook for beginners, vegan longevity diet, so vegan in 5 [hardcover] 4 books collection set (9123767995)
- structured rank1: [E] So Vegan In 5 (1788701232)

### structured minus plain: largest_gains, query41130
Query: flat sliders see-through
NDCG difference: +0.5132
- plain rank1: [I] Felt Furniture Sliders Hardwood Floors X-PROTECTOR – 16 PCS 3 1/2" Furniture Sliders – Heavy Duty Felt Sliders Hard Surfaces – Move Your Furniture Easily & Safely! (B07QTY2HC3)
- structured rank1: [E] Cambridge Select Women's Open Toe Single Band Clear See-Through Slip-On Flat Slide Sandal,8 B(M) US,Orange (B07PZ8S64L)

### structured minus ling_doc2query: largest_losses, query107558
Query: utv flag
NDCG difference: -0.5208
- ling_doc2query rank1: [E] Forever Wave USA Subdued Thin Blue Line Punisher Flag (B074BC9YD1)
- structured rank1: [C] Tusk UTV Horizontal Flag Mount 1 3/4"-1 7/8" - Fits: Arctic Cat WILDCAT 1000i H.O. 2012-2016 (B003AYUXGE)

### structured minus ling_doc2query: largest_losses, query10864
Query: asus predator monitor
NDCG difference: -0.4936
- ling_doc2query rank1: [E] Acer Predator XB271HU bmiprz 27" WQHD (2560x1440) NVIDIA G-SYNC IPS Monitor, (Display Port & HDMI Port, 144Hz), Black (B06ZXZ3QBD)
- structured rank1: [S] Asus 24" Full HD 1ms 144Hz DP HDMI FreeSync/Adaptive Sync Eye Care eSports Gaming Monitor Model MG248QR (B075RHSR4H)

### structured minus ling_doc2query: largest_losses, query1634
Query: 10x10 white backdrop
NDCG difference: -0.4833
- ling_doc2query rank1: [E] ePhoto Photo Studio Portrait 100% Cotton 10' x 10' White Muslin Backdrop Background Muslin New 1010W (B00I86N6HC)
- structured rank1: [S] White Backdrop Curtain for Parties White Backdrop Drapes for Weddings Baby Shower Birthday Party Photo Videos Photography Wrinkle Free Backdrop 2 Panels 5ft x 10ft Polyester Fabric (B07VK99SXT)

### structured minus ling_doc2query: largest_losses, query81906
Query: police women dvd season 1
NDCG difference: -0.4645
- ling_doc2query rank1: [E] Police Woman - The Complete First Season (B000E1EHPY)
- structured rank1: [S] Police Woman: Complete Second Season (B0062VL4YW)

### structured minus ling_doc2query: largest_losses, query63338
Query: lol all star bbs
NDCG difference: -0.4549
- ling_doc2query rank1: [E] L.O.L. Surprise All-Star B.B.s Sports Series 1 Baseball Sparkly Dolls with 8 Surprises (B085B22DY6)
- structured rank1: [S] L.O.L. Surprise! Party Supplies, Party Favors Collection - Pet Series Surprise Ball (LOL2018) (B07QQTZR3Q)

### structured minus ling_doc2query: largest_gains, query13584
Query: bath and body works air plug in
NDCG difference: +0.5856
- ling_doc2query rank1: [C] Bath and Body Works 4 Pack Linen and Lavender Wallflowers Fragrance Refill. 0.8 fl oz. (B07MQ7J7F5)
- structured rank1: [E] Bath&body Works White Wallflowers Pluggable Home Fragrance Diffuser Pack of 2 (B00W0IKK50)

### structured minus ling_doc2query: largest_gains, query74803
Query: nutri bullet 900
NDCG difference: +0.5100
- ling_doc2query rank1: [S] Ninja Nutri Pro Compact Personal Blender, with 18 Oz. and 24 Oz. To Go Cups, in a Black and Silver Finish (B00Y2U1QUM)
- structured rank1: [E] NutriBullet Pro - 9 Pc (B014ECJA2S)

### structured minus ling_doc2query: largest_gains, query106097
Query: twin xl bedding beach
NDCG difference: +0.5084
- ling_doc2query rank1: [S] Lush Decor Sealife Fish Ocean Wave Reversible 2 Piece Blue Quilt Bedding Set, Twin (B00O6465J6)
- structured rank1: [E] Greenland Home Maui Quilt Set, 2-Piece Twin/Twin XL, Multi (B01BDBQWB0)

### structured minus ling_doc2query: largest_gains, query24036
Query: cavalli man shoes size 9.5
NDCG difference: +0.4953
- ling_doc2query rank1: [I] ROBERTO CAVALLI 6609 Trainers Men White - 9.5 - Low Top Trainers Shoes (B07PB68CF5)
- structured rank1: [E] Roberto Cavalli 6284 Pony Hair Lace-up Bloucher (9.5) (B01HE9K1LW)

### structured minus ling_doc2query: largest_gains, query84862
Query: quadlock case iphone 11
NDCG difference: +0.4468
- ling_doc2query rank1: [I] Quad Lock Case for iPhone 11 Pro (B081J2MLK9)
- structured rank1: [E] Quad Lock Case for iPhone 11 (B081H6STBJ)

### hybrid_structured minus e5: largest_losses, query44158
Query: gameboy advance sp
NDCG difference: -0.6263
- e5 rank1: [E] Nintendo Game Boy Advance SP - Cobalt (Renewed) (B07KPNSR52)
- hybrid_structured rank1: [C] Wall Charger for Nintendo Gameboy DS Advance SP GBA, Power Charger Compatible with Nintendo Gameboy Advance SP (GBA SP) / Compatible with Nintendo Original Console (B07Z2TG635)

### hybrid_structured minus e5: largest_losses, query39932
Query: female bounty hunter costume
NDCG difference: -0.6056
- e5 rank1: [E] LOL The Bounty Hunter Miss Fortune Cosplay Costume Suits Custom Made Clothing (Female XXL) (B07PKSYTRZ)
- hybrid_structured rank1: [I] Bail Enforcement Agent Halloween Costume Shirt Bounty Hunter (B07J68Z4V9)

### hybrid_structured minus e5: largest_losses, query28565
Query: contact lens
NDCG difference: -0.5862
- e5 rank1: [E] Clinical Manual of Contact Lenses (1496397797)
- hybrid_structured rank1: [C] 2 in 1 Double Sided Portable Glasses Case and Contact Lens Case (Black) (B07SDC5MSQ)

### hybrid_structured minus e5: largest_losses, query11619
Query: b'one cup coffee dr'
NDCG difference: -0.5738
- e5 rank1: [E] Andaz Press 11oz. Coffee Mug Gag Gift, Doctor Because Badass Isn't an Official Title, 1-Pack (B079ZZP6B9)
- hybrid_structured rank1: [I] Two Rivers Coffee Flavored Coffee Pods Compatible with Keurig K Cup Brewers, Assorted Variety Pack Flavored Coffee, 40 Count (B00N0JIACS)

### hybrid_structured minus e5: largest_losses, query28627
Query: control board aux mini split
NDCG difference: -0.5593
- e5 rank1: [I] Gree 24,000 BTU 18 SEER Ceiling Cassette Ductless Mini Split Air Conditioner Heat Pump 208/230V (B0108SG73C)
- hybrid_structured rank1: [I] Klimaire DIY 12,000 BTU 20 SEER Mini Split Heat Pump Air Conditioner w/ 25-ft Pre-Charged Installation Kit Wi-fi - 115V (B08W28JBM5)

### hybrid_structured minus e5: largest_gains, query54900
Query: iron maiden hammer
NDCG difference: +0.5693
- e5 rank1: [I] Metal Hammer Iron Maiden - Final Frontier Mini Poster - 28.5x21cm (B00VXURZJU)
- hybrid_structured rank1: [E] METAL HAMMER MAGAZINE IRON MAIDEN MAIDEN ENGLAND (B078LRZSVC)

### hybrid_structured minus e5: largest_gains, query20010
Query: brother tn730 high yield black toner
NDCG difference: +0.5485
- e5 rank1: [I] Brother Genuine TN770 2-Pack Super High Yield Black Toner Cartridge with Approximately 4,500 Page Yield/Cartridge (B07VGV5438)
- hybrid_structured rank1: [E] Valuetoner Compatible Toner Cartridge Replacement for Brother TN760 TN-760 TN730 TN-730 High Yield for HL-L2350DW DCP-L2550DW HL-L2395DW Hl-L2390DW HL-L2370DW Printer (2 Black) (B07TLL8F94)

### hybrid_structured minus e5: largest_gains, query69159
Query: milt jackson ‎bebop lp
NDCG difference: +0.5000
- e5 rank1: [I] The Milt Jackson Quartet [Vinyl] (B000000XZQ)
- hybrid_structured rank1: [E] Miles Davis and Milt Jackson Quintet/Sextet [Vinyl] (B000000Y0I)

### hybrid_structured minus e5: largest_gains, query70293
Query: mom son daughter shirt
NDCG difference: +0.4760
- e5 rank1: [S] I Am A Lucky Son I Have A Crazy Mom TShirt Gifts T-Shirt (B07S8GLJC7)
- hybrid_structured rank1: [E] OK But First Coffee - Milk Mom & Son / Daughter Matching Set Mom & Baby Shirts Mom Black Medium / Baby Black Newborn (0-3M) (B07B486BR5)

### hybrid_structured minus e5: largest_gains, query5533
Query: 941m mercury
NDCG difference: +0.4750
- e5 rank1: [I] Mercury (B07PC4CL7W)
- hybrid_structured rank1: [E] Champion QL77CC 941M Marine 6 Pack (B0777PXCG5)
