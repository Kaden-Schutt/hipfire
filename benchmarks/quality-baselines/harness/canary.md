# Canary fixture — harness output reproducibility guard

11 wikitext-2 sequences (10 short, 1 long-ctx) with expected per-sequence
KLDs. Run before each eval; KLD divergence beyond per-sequence tolerance
aborts the eval with "harness regressed" — distinct from "reference
replaced" (caught by SHA256 in `manifest.json`).

**Purpose:** detect regressions in `eval_hipfire.rs` (kernel changes, fp
accumulator changes, prompt-handling drift) — not reference identity
changes. See plan §"Reference-drift canary (clarified)".

**Candidate model for canary:** `qwen3.5-9b.mq4` (canonical hipfire MQ4-uniform
9B; available locally at `~/.hipfire/models/qwen3.5-9b.mq4`). Candidate is
fixed across all canary runs; ref is the same as the bulk-eval ref.

## Sequences

Numbered s1–s11. Source: wikitext-2 train, distinct paragraphs picked for
character/topic diversity. Token counts measured with Qwen3.5/3.6 tokenizer.

| ID  | Length | Source description |
|-----|-------:|--------------------|
| s1   | ~ 115 | biography opening |
| s2   | ~ 171 | historical-events prose |
| s3   | ~ 196 | list-heavy with named entities |
| s4   | ~ 197 | geographic / coordinates |
| s5   | ~ 305 | scientific / technical terminology |
| s6   | ~ 235 | cultural / proper nouns + transliterations |
| s7   | ~ 348 | narrative / plot-summary style |
| s8   | ~ 385 | sports / statistics / dates |
| s9   | ~ 498 | long argumentative prose |
| s10  | ~ 464 | mixed prose + tabular |
| s11  | ~1695 | near-2K-token long-context fixture (M8) |

(Lengths are targets; actual lengths land in the +/-10% range after token
count via the pinned tokenizer.)

## Sequence text

The actual text bytes are committed below (within markdown code blocks
delimited by `<!-- s1 -->` ... `<!-- /s1 -->` markers so the harness can
extract them programmatically).

<!-- s1 -->
It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 .
<!-- /s1 -->

<!-- s2 -->
= = = Depictions of children = = = 

 Barker 's sketches , drawings , and paintings of children were given to friends or to the parents of the subjects , donated to charitable institutions and church sponsored events , or exhibited through various art organizations . She illustrated magazine covers , dust jackets , and produced series of postcards for Raphael Tuck and other publishers such as Picturesque Children of the Allies ( 1915 ) , Seaside Holidays ( 1918 ) , and Shakespeare 's Boy and Girl Characters ( 1917 , 1920 ) . Her own Old Rhymes for All Times ( 1928 ) and The Lord of the Rushie River ( 1938 ) , a tale about a girl who lives among swans on a riverbank , were critically well received .
<!-- /s2 -->

<!-- s3 -->
= = Music video = = 

 The accompanying music video begins with a shot of an empty street , followed by clips of disadvantaged and poorer members of society going about their daily activities . Two men play dominoes on a wooden crate outside a building , a gang make fun of an elderly man hanging newspapers outside his store and an obese woman walks down the street . Clips of Carey leaning against a wall and sitting on some steps looking on at what is happening are shown . As the first chorus begins , everyone starts to dance joyfully in the street and help those in need . A gospel choir comes out of one of the buildings as the street becomes more crowded with people of all ages and backgrounds rejoicing and getting along with each other . One of the shops in the background has a neon light outside the entrance which says " Jesus Saves " . 

 = = Track listings = = 

 " There 's Got to Be a Way " ( Original album version ) – 4 : 52
<!-- /s3 -->

<!-- s4 -->
Near the end of World War I , the Erzherzog Karl @-@ class battleships were handed over to the newly formed State of Slovenes , Croats and Serbs but Erzherzog Ferdinand Max was later transferred to Great Britain as a war reparation . She was later broken up for scrap in 1921 . 

 = Ancient Egyptian deities = 

 Ancient Egyptian deities are the gods and goddesses worshipped in ancient Egypt . The beliefs and rituals surrounding these gods formed the core of ancient Egyptian religion , which emerged sometime in prehistory . Deities represented natural forces and phenomena , and the Egyptians supported and appeased them through offerings and rituals so that these forces would continue to function according to maat , or divine order . After the founding of the Egyptian state around 3100 BC , the authority to perform these tasks was controlled by the pharaoh , who claimed to be the gods ' representative and managed the temples where the rituals were carried out .
<!-- /s4 -->

<!-- s5 -->
Prayer and private offerings are generally called " personal piety " : acts that reflect a close relationship between an individual and a god . Evidence of personal piety is scant before the New Kingdom . Votive offerings and personal names , many of which are theophoric , suggest that commoners felt some connection between themselves and their gods . But firm evidence of devotion to deities became visible only in the New Kingdom , reaching a peak late in that era . Scholars disagree about the meaning of this change — whether direct interaction with the gods was a new development or an outgrowth of older traditions . Egyptians now expressed their devotion through a new variety of activities in and around temples . They recorded their prayers and their thanks for divine help on stelae . They gave offerings of figurines that represented the gods they were praying to , or that symbolized the result they desired ; thus a relief image of Hathor and a statuette of a woman could both represent a prayer for fertility . Occasionally , a person took a particular god as a patron , dedicating his or her property or labor to the god 's cult . These practices continued into the latest periods of Egyptian history . These later eras saw more religious innovations , including the practice of giving animal mummies as offerings to deities depicted in animal form , such as the cat mummies given to the feline goddess Bastet . Some of the major deities from myth and official religion were rarely invoked in popular worship , but many of the great state gods were important in popular tradition .
<!-- /s5 -->

<!-- s6 -->
= = = Planning system = = = 

 The planning system is critical to the viability and operation of GA aerodromes . With many cities lacking scheduled air transport services between them , and with GA access to commercial airports becoming increasingly difficult and expensive , a viable network of aerodromes supporting GA air transport operations is regarded as an important national issue . However , there is no unified national planning policy specific to GA aerodromes , and planning decisions relating to these are based on local issues that are not required to consider the national impact . Because aircraft are excluded from noise control legislation , the only recourse for people affected by aircraft noise is through the planning process , and this issue is the principal factor on which the majority of planning decisions relating to GA land use are made . GA is a specialist subject often unfamiliar to Local Planning Authorities , and most planning decisions relating to GA either refuse permission , or grant it with restrictive conditions . Little Gransden is just one example of a GA airfield required to comply with planning restrictions on the number of movements permitted , thereby inhibiting further development . Such restrictions , if poorly conceived , can make GA operations unviable or even unsafe .
<!-- /s6 -->

<!-- s7 -->
= = = Types of ylides = = = 

 Many types of ylides can be prepared with various functional groups both on the anionic carbon center and on the sulfur . The substitution pattern can influence the ease of preparation for the reagents ( typically from the sulfonium halide , e.g. trimethylsulfonium iodide ) and overall reaction rate in various ways . The general format for the reagent is shown on the right . 

 Use of a sulfoxonium allows more facile preparation of the reagent using weaker bases as compared to sulfonium ylides . ( The difference being that a sulfoxonium contains a doubly bonded oxygen whereas the sulfonium does not . ) The former react slower due to their increased stability . In addition , the dialkylsulfoxide by @-@ products of sulfoxonium reagents are greatly preferred to the significantly more toxic , volatile , and odorous dialkylsulfide by @-@ products from sulfonium reagents . 

 The vast majority of reagents are monosubstituted at the ylide carbon ( either R1 or R2 as hydrogen ) . Disubstituted reagents are much rarer but have been described : 

 If the ylide carbon is substituted with an electron @-@ withdrawing group ( EWG ) , the reagent is referred to as a stabilized ylide . These , similarly to sulfoxonium reagents , react much slower and are typically easier to prepare . These are limited in their usefulness as the reaction can become prohibitively sluggish : examples involving amides are widespread , with many fewer involving esters and virtually no examples involving other EWG 's . For these , the related Darzens reaction is typically more appropriate .
<!-- /s7 -->

<!-- s8 -->
The novel has also been adapted for the stage , by Jorge Alí Triana and his daughter Veronica Triana , directed by Jorge Triana : the play was put on ( in Spanish , but with simultaneous translation to English ) at Repertorio Español ( www.repertorio.org / chivo ) in New York in 2003 ; and the production moved to Lima in 2007 . A feature of the novel 's stage version is that the same actor plays both Agustin Cabral and Rafael Trujillo . For reviewer Bruce Weber , this makes the point " that Trujillo 's control of the nation depended on gutless collaborators " . 

 = Charles Eaton ( RAAF officer ) = 

 Charles Eaton , OBE , AFC ( 21 December 1895 – 12 November 1979 ) was a senior officer and aviator in the Royal Australian Air Force ( RAAF ) , who later served as a diplomat . Born in London , he joined the British Army upon the outbreak of World War I and saw action on the Western Front before transferring to the Royal Flying Corps in 1917 . Posted as a bomber pilot to No. 206 Squadron , he was twice captured by German forces , and twice escaped . Eaton left the military in 1920 and worked in India until moving to Australia in 1923 . Two years later he joined the RAAF , serving initially as an instructor at No. 1 Flying Training School . Between 1929 and 1931 , he was chosen to lead three expeditions to search for lost aircraft in Central Australia , gaining national attention and earning the Air Force Cross for his " zeal and devotion to duty " . 

 In 1939 , on the eve of World War II , Eaton became the inaugural commanding officer of No.
<!-- /s8 -->

<!-- s9 -->
In 2010 , a European science team investigated the star using the CORALIE spectrograph and collected seventeen spectra of WASP @-@ 44 . From the spectra , radial velocity measurements were extrapolated . Analysis of collected CORALIE data ruled out the possibility that the detected radial velocity was caused by the blended spectrum of a spectroscopic binary star , supporting the possibility that the body orbiting WASP @-@ 44 was indeed a planet , designated WASP @-@ 44b . 

 The Leonhard Euler Telescope at La Silla Observatory in Chile was used to follow up on the discovery circling WASP @-@ 44 , searching for a point at which the planet transited , or crossed in front of , its host star . One transit was detected . 

 WASP @-@ 44 , its recently discovered planet , the planets orbiting WASP @-@ 45 and WASP @-@ 46 , and a discussion exploring the validity of the common assumption amongst scientists that closely orbiting Hot Jupiter planets have highly circular orbits unless proven otherwise , were reported in a single discovery paper that was published on May 17 , 2011 by the Royal Astronomical Society . The paper was submitted to the Monthly Notices of the Royal Astronomical Society on May 16 , 2011 . 

 = = Characteristics = = 

 WASP @-@ 44 is a G @-@ type star ( the same class of star as the Sun ) that is located in the Cetus constellation . WASP @-@ 44 has a mass that is 0 @.@ 951 times that of the Sun . In terms of size , WASP @-@ 44 has a radius that is 0 @.@ 927 times that of the Sun . WASP @-@ 44 has an effective temperature of 5410 K , cooler than the Sun . However , the star is metal @-@ rich with relation to the Sun . Its measured metallicity is [ Fe / H ] = 0 @.@ 06 , or 1 @.@ 148 times that the amount of iron found in the Sun . WASP @-@ 44 's chromosphere ( outermost layer ) is not active . The star also does not rotate at a high velocity .
<!-- /s9 -->

<!-- s10 -->
= = = Ziltoid the Omniscient and hiatus ( 2006 – 2008 ) = = = 

 Townsend withdrew from touring to spend time with his family . From home , Townsend completed his second solo ambient album , The Hummer , releasing it exclusively on his website in November 2006 . 

 In May 2007 , Townsend released Ziltoid the Omniscient , a tongue @-@ in @-@ cheek rock opera about the eponymous fictional alien . This was truly a solo album ; he programmed the drums using Drumkit from Hell , a software drum machine that uses samples recorded by Tomas Haake of Meshuggah and played all other instruments himself . Shortly after the album 's release , Townsend announced that he no longer planned to tour or make albums with Strapping Young Lad or the Devin Townsend Band . He explained that he was " burnt out on travelling , touring , and self promotion " and wished to do production work , write albums , and spend time with his family without the stress of interviews or touring . 

 In 2008 , Townsend lent his voice to characters in several episodes of the Adult Swim cartoon Metalocalypse ( see Musician cameos in Metalocalypse for more ) . The original character design for Pickles the Drummer , one of the series ' main characters , bore a striking resemblance to Townsend . The series ' co @-@ creator Brendon Small acknowledged the similarity , and altered the design before the series began . " We made sure he didn 't look like Devin Townsend . We gave him the goatee and the dreadover so he wouldn 't look like that . " 

 = = = Devin Townsend Project ( 2008 – 2012 ) = = = 

 After removing himself from the music industry , Townsend cut his trademark hair off and gave up drinking and smoking . Townsend found it " disconcerting " that he had difficulty writing music without drugs , and that he had trouble identifying his purpose as a musician . He spent a year producing albums in absence of writing , but found it unrewarding and decided to " pick up the guitar and just write " . This began a period of " self discovery " where he learned " how to create without drugs " .
<!-- /s10 -->

<!-- s11 -->
= Michael Jordan = 

 Michael Jeffrey Jordan ( born February 17 , 1963 ) , also known by his initials , MJ , is an American retired professional basketball player . He is also a businessman , and principal owner and chairman of the Charlotte Hornets . Jordan played 15 seasons in the National Basketball Association ( NBA ) for the Chicago Bulls and Washington Wizards . His biography on the NBA website states : " By acclamation , Michael Jordan is the greatest basketball player of all time . " Jordan was one of the most effectively marketed athletes of his generation and was considered instrumental in popularizing the NBA around the world in the 1980s and 1990s . 

 Jordan played three seasons for coach Dean Smith at the University of North Carolina . He was a member of the Tar Heels ' national championship team in 1982 . Jordan joined the NBA 's Chicago Bulls in 1984 as the third overall draft pick . He quickly emerged as a league star , entertaining crowds with his prolific scoring . His leaping ability , demonstrated by performing slam dunks from the free throw line in slam dunk contests , earned him the nicknames " Air Jordan " and " His Airness " . He also gained a reputation for being one of the best defensive players in basketball . In 1991 , he won his first NBA championship with the Bulls , and followed that achievement with titles in 1992 and 1993 , securing a " three @-@ peat " . Although Jordan abruptly retired from basketball before the beginning of the 1993 – 94 NBA season to pursue a career in baseball , he returned to the Bulls in March 1995 and led them to three additional championships in 1996 , 1997 , and 1998 , as well as a then @-@ record 72 regular @-@ season wins in the 1995 – 96 NBA season . Jordan retired for a second time in January 1999 , but returned for two more NBA seasons from 2001 to 2003 as a member of the Wizards . 

 Jordan 's individual accolades and accomplishments include five Most Valuable Player ( MVP ) Awards , ten All @-@ NBA First Team designations , nine All @-@ Defensive First Team honors , fourteen NBA All @-@ Star Game appearances , three All @-@ Star Game MVP Awards , ten scoring titles , three steals titles , six NBA Finals MVP Awards , and the 1988 NBA Defensive Player of the Year Award . Among his numerous accomplishments , Jordan holds the NBA records for highest career regular season scoring average ( 30 @.@ 12 points per game ) and highest career playoff scoring average ( 33 @.@ 45 points per game ) . In 1999 , he was named the greatest North American athlete of the 20th century by ESPN , and was second to Babe Ruth on the Associated Press 's list of athletes of the century . Jordan is a two @-@ time inductee into the Basketball Hall of Fame , having been enshrined in 2009 for his individual career , and again in 2010 as part of the group induction of the 1992 United States men 's Olympic basketball team ( " The Dream Team " ) . He became a member of the FIBA Hall of Fame in 2015 . 

 Jordan is also known for his product endorsements . He fueled the success of Nike 's Air Jordan sneakers , which were introduced in 1985 and remain popular today . Jordan also starred in the 1996 feature film Space Jam as himself . In 2006 , he became part @-@ owner and head of basketball operations for the then @-@ Charlotte Bobcats , buying a controlling interest in 2010 . In 2015 , as a result of the increase in value of NBA franchises , Jordan became the first billionaire NBA player in history and the world 's second @-@ richest African @-@ American . 

 = = Early years = = 

 Jordan was born in Brooklyn , New York , the son of Deloris ( née Peoples ) , who worked in banking , and James R. Jordan , Sr. , an equipment supervisor . His family moved to Wilmington , North Carolina , when he was a toddler . 

 Jordan is the fourth of five children . He has two older brothers , Larry Jordan and James R. Jordan , Jr . , one older sister , Deloris , and a younger sister , Roslyn . Jordan 's brother James retired in 2006 as the Command Sergeant Major of the 35th Signal Brigade of the XVIII Airborne Corps in the U.S. Army . 

 = = High school career = = 

 Jordan attended Emsley A. Laney High School in Wilmington , where he anchored his athletic career by playing baseball , football , and basketball . He tried out for the varsity basketball team during his sophomore year , but at 5 ' 11 " ( 1 @.@ 80 m ) , he was deemed too short to play at that level . His taller friend , Harvest Leroy Smith , was the only sophomore to make the team . 

 Motivated to prove his worth , Jordan became the star of Laney 's junior varsity squad , and tallied several 40 @-@ point games . The following summer , he grew four inches ( 10 cm ) and trained rigorously . Upon earning a spot on the varsity roster , Jordan averaged about 20 points per game over his final two seasons of high school play . As a senior , he was selected to the McDonald 's All @-@ American Team after averaging a triple @-@ double : 29 @.@ 2 points , 11 @.@ 6 rebounds , and 10 @.@ 1 assists . 

 Jordan was recruited by numerous college basketball programs , including Duke , North Carolina , South Carolina , Syracuse , and Virginia . In 1981 , Jordan accepted a basketball scholarship to North Carolina , where he majored in cultural geography . 

 = = College career = = 

 As a freshman in coach Dean Smith 's team @-@ oriented system , he was named ACC Freshman of the Year after he averaged 13 @.@ 4 points per game ( ppg ) on 53 @.@ 4 % shooting ( field goal percentage ) . He made the game @-@ winning jump shot in the 1982 NCAA Championship game against Georgetown , which was led by future NBA rival Patrick Ewing . Jordan later described this shot as the major turning point in his basketball career . During his three seasons at North Carolina , he averaged 17 @.@ 7 ppg on 54 @.@ 0 % shooting , and added 5 @.@ 0 rebounds per game ( rpg ) . He was selected by consensus to the NCAA All @-@ American First Team in both his sophomore ( 1983 ) and junior ( 1984 ) seasons . After winning the Naismith and the Wooden College Player of the Year awards in 1984 , Jordan left North Carolina one year before his scheduled graduation to enter the 1984 NBA draft . The Chicago Bulls selected Jordan with the third overall pick , after Hakeem Olajuwon ( Houston Rockets ) and Sam Bowie ( Portland Trail Blazers ) . One of the primary reasons why Jordan was not drafted sooner was because the first two teams were in need of a center . However , the Trail Blazers general manager Stu Inman contended that it was not a matter of drafting a center , but more a matter of taking Sam Bowie over Jordan , in part because Portland already had a guard with similar skills to Jordan , Clyde Drexler . ESPN , citing Bowie 's injury @-@ laden college career , named the Blazers ' choice of Bowie as the worst draft pick in North American professional sports history . Jordan returned to North Carolina to complete his degree in 1986 . 

 = = Professional career = =
<!-- /s11 -->
## Expected per-sequence KLDs (TBD — fills in after Step 4)

| Seq | NORMALIZE_PROMPT=0 | NORMALIZE_PROMPT=1 | Tolerance |
|-----|-------------------:|-------------------:|----------:|
| s1  |               TBD  |               TBD  |   ±15%    |
| s2  |               TBD  |               TBD  |   ±15%    |
| s3  |               TBD  |               TBD  |   ±15%    |
| s4  |               TBD  |               TBD  |   ±15%    |
| s5  |               TBD  |               TBD  |   ±15%    |
| s6  |               TBD  |               TBD  |   ±15%    |
| s7  |               TBD  |               TBD  |   ±15%    |
| s8  |               TBD  |               TBD  |   ±15%    |
| s9  |               TBD  |               TBD  |   ±15%    |
| s10 |               TBD  |               TBD  |   ±15%    |
| s11 |               TBD  |               TBD  |   ±15%    |

The "both NORMALIZE settings" hedge addresses Gemini m2 — if KLDs diverge
meaningfully between the two, that's a finding (and the eval-mode default
of OFF is revisited).

## How the harness uses this

1. Eval harness extracts s1–s11 from the markers above.
2. Tokenizes each via the candidate's tokenizer.
3. Runs candidate forward (already done — same as bulk eval).
4. Computes per-sequence KLD against the corresponding bytes of the
   BF16 reference.
5. Compares to the table above. Per-sequence delta within tolerance →
   pass. Any sequence outside tolerance → fail with which sequence(s)
   regressed.
