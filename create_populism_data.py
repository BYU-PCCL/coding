import pandas as pd
import os

# Make directories in data directory for
for dirname in [
    "badelite",
    "badeliterationale",
    "badelite1",
    "badelite2",
    "goodpeople",
    "goodpeoplerationale",
    "goodpeople1",
    "goodpeople2",
    "goodpeople3",
    "goodpeople4",
]:
    os.makedirs(f"data/{dirname}", exist_ok=True)


# Get data
jdf = pd.read_csv("data/JOP - Combined Populism data.csv", encoding="latin-1")
jdf = jdf["QID badelite goodpeople TEXT populism".split()]
jdf = jdf.dropna()
jdf

# Bad elite
bdf = jdf["badelite TEXT".split()]
bdf["ground_truth"] = bdf["badelite"].apply(lambda x: "true" if x else "false")

badelite1 = (
    lambda text: f"""The following texts are answers to the question "What is the country's most serious problem"? Each may or may not describe a "bad elite", or an agent or group that can be considered socially or politically powerful, such as "Congress" or "the health care industry". Please mark each "true" if it does describe a bad elite and "false" if it does not describe a bad elite:

"The elite wealthy; The income inequality in this country has been rapidly accelerating for the past several decades.": true
"Corporate business; Businesses are interested in making the biggest profits at the expense of their workers and consumers.": true
"The Luciferians; Duality is not reality; we are one.": false
"Culture and parents; If a student does not value education, no amount of money is going to get that student to learn.": false
"{text}":"""
)
bdf["prompt"] = bdf["TEXT"].apply(badelite1)
bdf["template_name"] = "badelite1"
bdf.to_pickle("data/badelite1/ds.pkl")


badelite2 = (
    lambda text: f"""The following texts are answers to the question "What is the country's most serious problem"? Each may or may not describe a "bad elite", or an agent or group that could have the power to cause political or economic change domestically. Please mark each "true" if it does describe a bad elite and "false" if it does not describe a bad elite:

"Bad economic decisions and priorities by elected officials; I would like to see elected officials strive to invest more money in education and infrastructure and enact trade policies more favorable to American companies.": true
"Extremist groups from different Arab countries; The groups of people have been indoctrinated to hate America. They don’t like the freedoms we have and they don’t like us. There people need to be brought to justice by any means necessary.": false
"Politicians; Well they are the ones leading our country in the wrong direction.": true
"Al Qaeda; They have very extreme views that involve killing people. We should kill them.": false
"{text}":"""
)
bdf["prompt"] = bdf["TEXT"].apply(badelite2)
bdf["template_name"] = "badelite2"
bdf.to_pickle("data/badelite2/ds.pkl")


# Good people
gdf = jdf["goodpeople TEXT".split()]
gdf["ground_truth"] = gdf["goodpeople"].apply(lambda x: "true" if x else "false")

goodpeople1 = (
    lambda text: f"""The following texts are answers to the question "What is the country's most serious problem"? Each may or may not describe a "clearly distinguished majority" (a separate entity; not simply humanity as a whole) that is hurt by the bad actor/bad elite. Please mark each "true" if it does describe such a majority and "false" if it does not describe a majority:

"Politicians; The politicians are the ones that are keeping social differences as they are, and creating chaos. If they would work for the people, the people would be better off.": true
"Humanity; We do not take care of the environment, we abuse it. We use too much gas. We should reduce our usage.": false
"Congressional Republicans; They are in the pockets of big business. They are responsible for the government shutdown and prevent any positive legislation from being passed to help the middle class.": true
"Humans; Many are unaware of the effect products and actions have on the environment. People need to be educated to become aware of the climate changes that are happening and could happen.": false
"{text}":"""
)
gdf["prompt"] = gdf["TEXT"].apply(goodpeople1)
gdf["template_name"] = "goodpeople1"
gdf.to_pickle("data/goodpeople1/ds.pkl")

goodpeople2 = (
    lambda text: f"""The following texts are answers to the question "What is the country's most serious problem"? Each may or may not ascribe positive moral attributes, either explicitly or implicitly, to the people; this can include being seen as the commoners, the regular guy, the workers, etc. Please mark each "true" if it does ascribe such positive attributes to the people and "false" if it does not ascribe such positive attributes to the people:

"Politicians and business owners; They have done very little to prevent the growing income gap between the average worker and those in control of America’s businesses.": true
"Hypersensitivity; It seems that everyone is offended by things they don’t agree with nowadays.": false
"Everyone is responsible; It is the responsibility of the entire human race to protect our planet.": false
"Big corporations; they commit fraud and become overly greedy at the expense of regular individuals like myself.": true
"{text}":"""
)
gdf["prompt"] = gdf["TEXT"].apply(goodpeople2)
gdf["template_name"] = "goodpeople2"
gdf.to_pickle("data/goodpeople2/ds.pkl")

goodpeople3 = (
    lambda text: f"""The following texts are answers to the question "What is the country's most serious problem"? Each may or may not mention an “us vs. them” mentality (including the 1% vs. the 99%). Please mark each "true" if it does mention an “us vs. them” mentality and "false" if it does not mention an “us vs. them” mentality:

"The upper 1 percent; They just want to keep as much money for themselves, so they use loopholes, bailouts, etc and in the long run it just hurts everybody else.": true
"Corporations; Big companies use the greatest amount of resources. We need policy limiting the damage they can do.": false
"Deadbeats, lazy people, and Obama; They use our money to work against us.": true
"For profit insurance companies; We’ve tried to follow the Swiss model, but it doesn’t work.": false
"{text}":"""
)
gdf["prompt"] = gdf["TEXT"].apply(goodpeople3)
gdf["template_name"] = "goodpeople3"
gdf.to_pickle("data/goodpeople3/ds.pkl")

goodpeople4 = (
    lambda text: f"""The following texts are answers to the question "What is the country's most serious problem"? Each may or may not identify the common people as the group fit to have control. Please mark each "true" if it does identify the common people as the group fit to have control and "false" if it does not identify the common people as the group fit to have control:

"Politicians; They only look out for their own self interest. I think the ordinary people should lead the country so even the little mans interests are taken into consideration.": true
"Declines in teachers’ ability and increasing classroom disruptions; Teachers aren’t teaching to the children’s abilities. They are teaching so that they can pass tests.": false
"Liberals; They generally do not uphold the traditional values of the masses but rather the interest of small select interest groups. We ought to uphold the traditional values of the people.": true
"Kids are getting dumber and we’re doing nothing about it; Parents should be held accountable.": false
"{text}":"""
)
gdf["prompt"] = gdf["TEXT"].apply(goodpeople4)
gdf["template_name"] = "goodpeople4"
gdf.to_pickle("data/goodpeople4/ds.pkl")

# Bad elite
bdf = jdf["badelite TEXT".split()]
bdf["ground_truth"] = bdf["badelite"].apply(lambda x: "true" if x else "false")

badelite = (
    lambda text: f"""The following texts are answers to the question "What is the country's most serious problem"? Each may or may not describe a "bad elite"; a bad elite is an elite agent, group, or actor that can be considered socially or politically powerful (such as "Congress" or "the health care industry") that is construed in the response as malicious. This bad elite must have the power to cause political or economic change domestically. Please mark each "true" if it does describe a bad elite and "false" if it does not describe a bad elite:


"Bad economic decisions and priorities by elected officials; I would like to see elected officials strive to invest more money in education and infrastructure and enact trade policies more favorable to American companies.": true
"Extremist groups from different Arab countries; The groups of people have been indoctrinated to hate America. They don’t like the freedoms we have and they don’t like us. There people need to be brought to justice by any means necessary.": false
"Politicians; Well they are the ones leading our country in the wrong direction.": true
"Al Qaeda; They have very extreme views that involve killing people. We should kill them.": false
"The elite wealthy; The income inequality in this country has been rapidly accelerating for the past several decades.": true
"Corporate business; Businesses are interested in making the biggest profits at the expense of their workers and consumers.": true
"The Luciferians; Duality is not reality; we are one.": false
"Culture and parents; If a student does not value education, no amount of money is going to get that student to learn.": false
"{text}":"""
)
bdf["prompt"] = bdf["TEXT"].apply(badelite)
bdf["template_name"] = "badelite"
bdf.to_pickle("data/badelite/ds.pkl")

# Good people
gdf = jdf["goodpeople TEXT".split()]
gdf["ground_truth"] = gdf["goodpeople"].apply(lambda x: "true" if x else "false")

goodpeople = (
    lambda text: f"""The following texts are answers to the question "What is the country's most serious problem"? Each may or may not describe "good people", i.e., a collective majority of the citizenry constituted by ordinary citizens who are seen positively that is hurt by the bad actor/bad elite. Please mark each "true" if it does describe such a "good people" and "false" if it does not describe such a "good people":

"Politicians; The politicians are the ones that are keeping social differences as they are, and creating chaos. If they would work for the people, the people would be better off.": true
"Humanity; We do not take care of the environment, we abuse it. We use too much gas. We should reduce our usage.": false
"Congressional Republicans; They are in the pockets of big business. They are responsible for the government shutdown and prevent any positive legislation from being passed to help the middle class.": true
"Humans; Many are unaware of the effect products and actions have on the environment. People need to be educated to become aware of the climate changes that are happening and could happen.": false
"Politicians; They only look out for their own self interest. I think the ordinary people should lead the country so even the little mans interests are taken into consideration.": true
"Declines in teachers’ ability and increasing classroom disruptions; Teachers aren’t teaching to the children’s abilities. They are teaching so that they can pass tests.": false
"Liberals; They generally do not uphold the traditional values of the masses but rather the interest of small select interest groups. We ought to uphold the traditional values of the people.": true
"Kids are getting dumber and we’re doing nothing about it; Parents should be held accountable.": false
"The upper 1 percent; They just want to keep as much money for themselves, so they use loopholes, bailouts, etc and in the long run it just hurts everybody else.": true
"Corporations; Big companies use the greatest amount of resources. We need policy limiting the damage they can do.": false
"Deadbeats, lazy people, and Obama; They use our money to work against us.": true
"For profit insurance companies; We’ve tried to follow the Swiss model, but it doesn’t work.": false
"Politicians and business owners; They have done very little to prevent the growing income gap between the average worker and those in control of America’s businesses.": true
"Hypersensitivity; It seems that everyone is offended by things they don’t agree with nowadays.": false
"Everyone is responsible; It is the responsibility of the entire human race to protect our planet.": false
"Big corporations; they commit fraud and become overly greedy at the expense of regular individuals like myself.": true
"{text}":"""
)
gdf["prompt"] = gdf["TEXT"].apply(goodpeople)
gdf["template_name"] = "goodpeople"
gdf.to_pickle("data/goodpeople/ds.pkl")

### Rationale

# Bad elite
bdf = jdf["badelite TEXT".split()]
bdf["ground_truth"] = bdf["badelite"].apply(lambda x: "true" if x else "false")

badeliterationale = (
    lambda text: f"""The following texts are answers to the question "What is the country's most serious problem"? Each may or may not describe a "bad elite", i.e., an agent, group, or actor that can be considered socially or politically powerful (such as "Congress" or "the health care industry") that is construed in the response as malicious. This bad elite must have the power to cause political or economic change domestically. Please mark each "true" if it does describe a bad elite and "false" if it does not describe a bad elite, providing a rationale for your answer:

text: "Bad economic decisions and priorities by elected officials; I would like to see elected officials strive to invest more money in education and infrastructure and enact trade policies more favorable to American companies."
rationale: Elected officials are blamed for having bad economic decisions and priorities, so a bad actor is identified. These officials are seen as elites and have the power to cause political or economic change domestically because they hold domestic political office.
bad elite: true

text: "Everyone is responsible; The health and welfare of the environment is the responsibility of everyone. We need to protect it as much as we can."
rationale: No individual actor is identified.
bad elite: false

text: "Extremist groups from different Arab countries; The groups of people have been indoctrinated to hate America. They don’t like the freedoms we have and they don’t like us. There people need to be brought to justice by any means necessary."
rationale: Foreign extremist groups are identified, so a bad actor is identified. They do not have the power to cause political or economic change domestically because they are a foreign entity.
bad elite: false

text: "Politicians; Well they are the ones leading our country in the wrong direction."
rationale: Politicians are blamed, so a bad actor is identified, and politicians are seen as elite and have power to cause political and economic change domestically.
bad elite: true

text: "Al Qaeda; They have very extreme views that involve killing people. We should kill them."
rationale: Al Qaeda is blamed, so a bad actor is identified. Al Qaeda doesn't have power to cause political or economic change domestically, however.
bad elite: false

text: "The elite wealthy; The income inequality in this country has been rapidly accelerating for the past several decades."
rationale: The elite wealthy is blamed, so a bad actor is identified, and it is a small, elite class with power to cause political change domestically.
bad elite: true

text: "Corporate business; Businesses are interested in making the biggest profits at the expense of their workers and consumers."
rationale: Corporate business is blamed, so a bad actor is identified, and it is an elite group with power to cause economic change domestically.
bad elite: true

text: "The Luciferians; Duality is not reality; we are one."
rationale: The Luciferians are blamed, so a bad actor is identified, but it is not seen as an elite, since the Luciferians are a small, obscure group of people.
bad elite: false

text: "Culture and parents; If a student does not value education, no amount of money is going to get that student to learn."
rationale: Culture and parents are blamed, so a bad actor is identified, but culture and parents are not seen as an elite class of society, since most adults are parents and everyone contributes to culture.
bad elite: false

text: "No particular group; It isn't a particular group. All groups of people need to recognize that intolerance is a problem and that open-mindedness is a solution."
rationale: No individual actor is identified.
bad elite: false

text: "{text}"
rationale:"""
)
bdf["prompt"] = bdf["TEXT"].apply(badeliterationale)
bdf["template_name"] = "badeliterationale"
bdf.to_pickle("data/badeliterationale/ds.pkl")

# Good people
gdf = jdf["goodpeople TEXT".split()]
gdf["ground_truth"] = gdf["goodpeople"].apply(lambda x: "true" if x else "false")

goodpeoplerationale = (
    lambda text: f"""The following texts are answers to the question "What is the country's most serious problem"? Each may or may not describe "good people", i.e., a collective majority of the citizenry (constituted by ordinary citizens who are seen positively) that is hurt by the bad actor/bad elite. Please mark each "true" if it does describe such a "good people" and "false" if it does not describe such a "good people", providing a rationale for your answer:

text: "Politicians; The politicians are the ones that are keeping social differences as they are, and creating chaos. If they would work for the people, the people would be better off."
rationale: The phrase "the people" suggests a good, virtuous people who are being hurt by a bad elite (politicians). 
bad elite: true

text: "Humanity; We do not take care of the environment, we abuse it. We use too much gas. We should reduce our usage."
rationale: The people are not seen as good and are being hurt by themselves, not an elite.
bad elite: false

text: "Congressional Republicans; They are in the pockets of big business. They are responsible for the government shutdown and prevent any positive legislation from being passed to help the middle class."
rationale: The "middle class" suggests a good group of ordinary folks who are being hurt by a bad elite (Congressional Republicans).
bad elite: true

text: "Humans; Many are unaware of the effect products and actions have on the environment. People need to be educated to become aware of the climate changes that are happening and could happen."
rationale: Humans are not being hurt by elites, but by themselves, and so the people are not seen as good.
bad elite: false

text: "Politicians; They only look out for their own self interest. I think the ordinary people should lead the country so even the little mans interests are taken into consideration."
rationale: The ordinary people are more fit to have control of the country than the elite (politicians) who are doing a bad job of things currently.
bad elite: true

text: "Declines in teachers’ ability and increasing classroom disruptions; Teachers aren’t teaching to the children’s abilities. They are teaching so that they can pass tests."
rationale: Teachers, who are ordinary people, are seen as having the fault for not teaching to the children’s abilities, as opposed to some bad elite doing it. So ordinary people aren't seen as good.
bad elite: false

text: "Liberals; They generally do not uphold the traditional values of the masses but rather the interest of small select interest groups. We ought to uphold the traditional values of the people."
rationale: Liberals are seen by many conservatives as being coastal elites who are uppity and self-interested, not caring about the masses. The traditional values of the people should be upheld, suggesting they are good.
bad elite: true

text: "Kids are getting dumber and we’re doing nothing about it; Parents should be held accountable."
rationale: Regular parents are seen as having the fault for diminishing educational outcomes, so the people aren't seen as good.
bad elite: false

text: "The upper 1 percent; They just want to keep as much money for themselves, so they use loopholes, bailouts, etc and in the long run it just hurts everybody else."
rationale: There are clear mentions of the 1% vs. the 99% in the text, suggesting malice in the former and virtue in the latter.
bad elite: true

text: "Corporations; Big companies use the greatest amount of resources. We need policy limiting the damage they can do."
rationale: Even though the companies are seen as bad, there is no reference to the people being good.
bad elite: false

text: "Deadbeats, lazy people, and Obama; They use our money to work against us."
rationale: An "us vs. them" mentality is clearly manifest.
bad elite: true

text: "For profit insurance companies; We’ve tried to follow the Swiss model, but it doesn’t work."
rationale: The insurance companies are seen as bad, but there is no reference to a virtuous people, either.
bad elite: false

text: "Politicians and business owners; They have done very little to prevent the growing income gap between the average worker and those in control of America’s businesses."
rationale: The "average worker" described suggests a virtuous, ordinary person who is being hurt by an elite.
bad elite: true

text: "Hypersensitivity; It seems that everyone is offended by things they don’t agree with nowadays."
rationale: The people are not seen as good, and instead of a bad elite, the problem is a cultural problem.
bad elite: false

text: "Everyone is responsible; It is the responsibility of the entire human race to protect our planet."
rationale: The people is not seen as virtuous, since all are described as having fault.
bad elite: false

text: "Big corporations; they commit fraud and become overly greedy at the expense of regular individuals like myself."
rationale: The "regular individuals" suggests that ordinary people are virtuous, and the "big corporations" are seen as bad elites hurting these good people.
bad elite: true

text: "{text}"
rationale:"""
)
gdf["prompt"] = gdf["TEXT"].apply(goodpeoplerationale)
gdf["template_name"] = "goodpeoplerationale"
gdf.to_pickle("data/goodpeoplerationale/ds.pkl")

# Populist 
pdf = jdf["populism TEXT".split()]
pdf["ground_truth"] = pdf["populism"].apply(lambda x: "true" if x else "false")

populist = (
    lambda text: f"""The following texts are answers to the question "What is the country's most serious problem"? Each may or may not be populist, in that it describes a "good people", or a virtuous majority of the citizenry (constituted by ordinary citizens who are seen positively), being hurt by a "bad elite", some particularly powerful minority of actors who have power to effect political or economic change domestically. Please mark each "true" if it is populist (including both "good people" and "bad elite") or "false" if it is not populist (excluding either "good people" or "bad elite" or both):


"Politicians; The politicians are the ones that are keeping social differences as they are, and creating chaos. If they would work for the people, the people would be better off.": true
"Humanity; We do not take care of the environment, we abuse it. We use too much gas. We should reduce our usage.": false
"Congressional Republicans; They are in the pockets of big business. They are responsible for the government shutdown and prevent any positive legislation from being passed to help the middle class.": true
"Humans; Many are unaware of the effect products and actions have on the environment. People need to be educated to become aware of the climate changes that are happening and could happen.": false
"Politicians; They only look out for their own self interest. I think the ordinary people should lead the country so even the little mans interests are taken into consideration.": true
"Declines in teachers’ ability and increasing classroom disruptions; Teachers aren’t teaching to the children’s abilities. They are teaching so that they can pass tests.": false
"Liberals; They generally do not uphold the traditional values of the masses but rather the interest of small select interest groups. We ought to uphold the traditional values of the people.": true
"Kids are getting dumber and we’re doing nothing about it; Parents should be held accountable.": false
"The upper 1 percent; They just want to keep as much money for themselves, so they use loopholes, bailouts, etc and in the long run it just hurts everybody else.": true
"Corporations; Big companies use the greatest amount of resources. We need policy limiting the damage they can do.": false
"Deadbeats, lazy people, and Obama; They use our money to work against us.": true
"Politicians and business owners; They have done very little to prevent the growing income gap between the average worker and those in control of America’s businesses.": true
"Hypersensitivity; It seems that everyone is offended by things they don’t agree with nowadays.": false
"Everyone is responsible; It is the responsibility of the entire human race to protect our planet.": false
"Big corporations; they commit fraud and become overly greedy at the expense of regular individuals like myself.": true
"Bad economic decisions and priorities by elected officials; I would like to see elected officials strive to invest more money in education and infrastructure and enact trade policies more favorable to American companies.": true
"Extremist groups from different Arab countries; The groups of people have been indoctrinated to hate America. They don’t like the freedoms we have and they don’t like us. There people need to be brought to justice by any means necessary.": false
"Politicians; Well they are the ones leading our country in the wrong direction.": true
"Al Qaeda; They have very extreme views that involve killing people. We should kill them.": false
"The elite wealthy; The income inequality in this country has been rapidly accelerating for the past several decades.": true
"Corporate business; Businesses are interested in making the biggest profits at the expense of their workers and consumers.": true
"The Luciferians; Duality is not reality; we are one.": false
"Culture and parents; If a student does not value education, no amount of money is going to get that student to learn.": false
"{text}":"""
)

pdf["prompt"] = pdf["TEXT"].apply(populist)
pdf["template_name"] = "populist"
pdf.to_pickle("data/populist/ds.pkl")
