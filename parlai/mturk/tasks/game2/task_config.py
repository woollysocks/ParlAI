#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}


task_config['frontend_version'] = 1

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Textual Entailment Game'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config[
    'hit_description'
] = 'As a writer you\'ll write 3 claims given a prompt and then as an evaluator, you will rank claims for a differnet prompt.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'textual entailment,nli,rte,entailment,sentence creation,writing,evaluation'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config[
    'task_description'
] = '''
<p>
	The <a href="https://cds.nyu.edu/" target="_blank">New York University Center for Data Science</a> is collecting your answers to study
	machine understanding of English. Thank you for your help! <i>[This is a pilot for a larger project]</i>
</p>

There are two phases in each HIT: the <b>writing phase</b> and the <b>evaluating phase</b>. Each HIT has 4 people on it, and you will be paired with one person. Because you're working in teams of 4, <b>please be warned that you might have to wait for someone else to complete their task during the HIT.</b> We are going to ensure that we pay you for these wait times as well, and not just for your active work time.<br><br>

<h4><u>Phase 1: Writing</u></h4>
<p> You, and your partner, will be shown a prompt that describes a situation or event. Using only this description and what you know about the world, please write</p>
<ul>
	<li>One claim that is <strong>definitely correct</strong><em>&nbsp;</em>about the situation or event in the prompt.</li>
	<li>One claim that is&nbsp;<strong>definitely incorrect</strong>&nbsp;about the situation or event in the prompt.</li>
	<li>One claim that is&nbsp;<strong>neither</strong> definitely correct nor incorrect about the situation or event in the prompt.</li>
</ul>
<font color="red">We encourage you to write claims that are creative and require non-trivial kinds of reasoning.</font> Your partner on this HIT will also be writing claims for this prompt and your bonuses will be based on how evaluators rank you versus your partner. We will show you the claims your partner wrote.<br>

<h4><u>Phase 2: Evaluating</u></h4>
<p> You will be shown a new prompt and the claims written by the 2 other people on this HIT. You will have to evaluate claims for each of the three labels---definitely correct, definitely incorrect, and neither---explained in the previous section. For each label and set of claims you will be asked to do the following,</p>
<ul>
	<li> Given the label and the two claims written by different people, <font color="red">decide if the claims fit the label and the prompt.</font></li>
	<li> Then, <font color="red">rank the claim in terms of creativity and complexity of reasoning.</font></li>
	<li> You have the option to leave a justification for your ranking in a text box. Anything you write here will be shown to your partner and to the writers of the claims at the end of the evaluation phase. We encourage you to use this text box generously!</li>
</ul>
Your bonuses will be based on your agreement with your partner on the rankings of the claims.<br>

<h4><u> Examples </u></h4>
<p>
Prompt: James Jolt moved to London this past summer.
</p>
<ul>
<li><p>
Label: Definitely correct<br>
Hypothesis: Some months ago James Jolt settled down in England.<br>
<em> Explanation: This example is made more interesting because there are some synonyms used, and the reader needs to know that London is in England.</em>
</p></li>
<li><p>
Label: Definitely incorrect<br>
Hypothesis: In spite the Brexit turmoil, James Jolt decided to keep his home and business in London, where they have been for the past decade.<br>
<em>Explanation: From the prompt we know that James just moved to London, and hasn't been there for 10 years. This hypothesis implies that James hasn't just moved to London without discussing moving, making it a more complicated and interesting example.</em>
</p></li>
<li><p>
Label: Neither (neither definitely correct nor definitely incorrect)<br>
Hypothesis: James Jolt recently moved across the pond.<br>
<em> Explanation: If James is moving from the United States, then this hyopthesis would be correct, but this may or may not be true given the prompt. Furthermore, the example is made more interesting because the reader needs to know "the pond" is sometimes used to refer to the Atlantic Ocean.</em>
</p></li>
</ul>

<p>
Prompt: Mother-in-law's tongue is one of the colloquial names for Sansevieria Trifasciata plant. It tolerates most light conditions and doesn't need frequent watering, making it an ideal house plant.
</p>
<ul>
<li><p>
Label: Definitely correct<br>
Hypothesis: Sansevieria Trifasciatais a good choice for beginners and those who don't want the hassle of a needy plant.<br>
<em> Explanation: This example is made more interesting by making the non-trivial inference that a plant that tolerates all kinds of light and is easy to water must be easy maintain and therefore good for beginners.</em>
</p></li>
<li><p>
Label: Definitely incorrect<br>
Hypothesis: Mother in law's tongue doesn't need lots of sunlight but is easy to underwater.<br>
<em>Explanation: The first part of this hypothesis is true and requires a step of inference from the prompt: tolerating lots of lighting conditions means it can tolerate low sunlight. The second part is a contradiction of the prompt and it uses some rephrasing by saying "underwater."</em>
</p></li>
<li><p>
Label: Neither (neither definitely correct nor definitely incorrect)<br>
Hypothesis: Mother in law's tongue is easy to maintain, growing quickly under most conditions.<br>
<em> Explanation: The premise doesn't state or directly imply that the plant will grow quickly, only that it will live in most conditions. The subtlety of this example makes it interesting.</em>
</p></li>
</ul>


<!-- TODO: COMPLETE THIS. ADD BONUS BREAKDOWN -->
<h4><u>Bonuses</u></h4>
<p>You will have opportunities to get bonuses in both phases,</p>
<ul>
	<li>Writing phase: your bonus is be based on your rank. For each claim that you rank higher than you partner, you will receive a $0.3 bonus. If the evaluators don't agree on a ranking for a label, then you and your partner will get $0.1.</li>
	<li>Evaluation phase: your bonus is based on your agreement with your partner. For a label, if you both rank the same claim first then you'll get a $0.1 bonus. If you don't agree, you don't receive a bonus.</li>
</ul>
<br>

<p>
For more questions please refer to this <a href="https://woollysocks.github.io/NLI-games-faqs/two-faq.html" target="_blank">FAQ page</a>.
<br>
Finally, if you are ready, please click "Accept" to start this task.
</p>
'''
