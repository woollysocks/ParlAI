#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
import threading
import random
import copy
import time
import itertools
import pandas as pd
from parlai.mturk.core.agents import TIMEOUT_MESSAGE

TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
#ToDo: add waiting message
# WAITING_MSG = 'Please wait while we match you with another worker...'

class OnboardingWorld(MTurkOnboardWorld):
    """Example onboarding world. Sends a message from the world to the
    worker and then exits as complete after the worker uses the interface
    """

    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = (
            "Welcome onboard! Before starting the HIT, please complete this onboarding.\n\nPlease read the instructions carefully and evaluate the following example by deciding if the claim fits the label and the prompt. In this onboarding you only have have to do evaluation, no writing. Thank you!"
        )
        self.mturk_agent.observe(ad)
        onboard_0_true = {
                'id': '<b>Prompt</b>',
                'text': 'He offered me one of the tiny Russian cigarettes he himself occasionally smoked.\
                    \n<b>Label</b>: Definitely correct\
                    \n\n<b>Claim</b>: He offered me one of the cigarettes that he sometime smoked.',
                    'task_data': {'respond_with_form':[], 'onboard':True}
            }
        onboard_0_false = {
                'id': '<b>Prompt</b>',
                'text': 'He offered me one of the tiny Russian cigarettes he himself occasionally smoked.\
                    \n<b>Label</b>: Neither definitely correct nor definitely incorrect\
                    \n\n<b>Claim</b>: He offered me one of the cigarettes that he sometime smoked.',
                    'task_data': {'respond_with_form':[], 'onboard':True}
            }

        onboard_1_true = {
                'id': '<b>Prompt</b>',
                'text': 'Drew was puzzled.\
                        \n<b>Label</b>: Definitely incorrect\
                        \n\n<b>Claim</b> Drew understood completely.',
                        'task_data': {'respond_with_form':[], 'onboard':True}
            }
        onboard_1_false = {
                'id': '<b>Prompt</b>',
                'text': 'Drew was puzzled.\
                    \n<b>Label</b>: Definitely correct\
                    \n\n<b>Claim</b> Drew understood completely.',
                        'task_data': {'respond_with_form':[], 'onboard':True}
            }

        onboard_2_true = {
                'id': '<b>Prompt</b>',
                'text': 'The non-stop lift takes 55 seconds from ground to the observatory, but offers spectacular views of the city and beyond!\
                    \n<b>Label</b>: Neither definitely correct nor definitely incorrect\
                    \n\n<b>Claim</b> The lift was designed to offer the spectacular views.',
                    'task_data': {'respond_with_form':[], 'onboard':True}
            }
        onboard_2_false = {
                'id': '<b>Prompt</b>',
                'text': 'The non-stop lift takes 55 seconds from ground to the observatory, but offers spectacular views of the city and beyond!\
                    \n<b>Label</b>: Definitely incorrect\
                    \n\n<b>Claim</b>: The lift was designed to offer the spectacular views.',
                    'task_data': {'respond_with_form':[], 'onboard':True}
            }

        onboard_options = [onboard_0_true, onboard_0_false, onboard_1_true, onboard_1_false, onboard_2_true, onboard_2_false]

        onboard_question = random.choice(onboard_options)
        onboard_answer = 'Valid'
        if onboard_question in onboard_options[1::2]:
            onboard_answer = 'Invalid'
        self.mturk_agent.observe(onboard_question)
        onboard_check = self.mturk_agent.act(onboard=onboard_answer)
        self.episodeDone = True
        return onboard_check

class MultiRoleAgentWorld(MTurkTaskWorld):
    """
    World to demonstrate workers with assymetric roles. This task amounts
    to three rounds and then an evaluation step. It is purposefully created
    as a task to demo multiple views and has no other purpose.
    """

    collector_agent_id = 'Moderator'

    def __init__(self, opt, task, mturk_agents):
        self.task = task
        self.mturk_agents = mturk_agents
        self.num_agents = len(mturk_agents)
        # We require even number of agents for this game
        assert self.num_agents % 2 == 0, "We have "+str(self.num_agents)+" agents. Need an even number of agents for this game."

        self.agents = [None for i in range(len(mturk_agents))]
        for agent in mturk_agents:
            num = int(agent.demo_role[-1]) # number form 'Person 1' etc.
            self.agents[num-1] = agent
        
        self.sets = {}
        for i in range(int(self.num_agents/2)):
            self.sets[i] = self.agents[i*2 : (i+1)*2]
        
        self.episodeDone = False
        self.max_meta_turns = 3
        self.meta_turn = 0
        self.interim_data = []
        self.turns = 0
        
        self.noflip = {'Claim 1':'Claim 1', 'Claim 2':'Claim 2'}
        self.yesflip = {'Claim 1':'Claim 2', 'Claim 2':'Claim 1'}
        self.writer_bonus = 0.3
        self.ranker_bonus = 0.1

        self.incomplete_message = {'id': '<b><big>INCOMPLETE</big></b>', 
                                'text': '<b><big> Your turn timed out and autosubmitted because these HITs require real-time interaction and we want to keep wait times low. </big></b>'}

    def parley(self):
        if self.meta_turn < self.max_meta_turns:
            if self.turns == 0:
                # Get prompt and limit to one sentence
                self.prompts = []
                for i in range(int(self.num_agents/2)):
                    squad_example = self.task.act()
                    premise = '\n'.join(squad_example['text'].split('\n')[:-1])
                    ## To-do: preprocess sentence segmentation. This is a janky fix ##
                    # premise = premise.split(".", 1)[0] + "."
                    prompt = {
                            'id': 'Prompt',
                            'text': premise,
                            'task_data': {'writing':True, 'feedback':False}
                        }
                    self.prompts.append(prompt)

                    # Assign writing prompts to sets of agents
                    for agent in self.sets[i]:
                        agent.observe(prompt)

                # Make copies of objects of workers, we need this to have
                # concurrent work on HITs
                self.writers_copy = self.agents.copy()
                self.evaluators_ent = self.agents.copy()
                self.evaluators_cont = []
                self.evaluators_neut = []
                self.keep_evaluator_status = []
                
                # Lists where we will store responses
                self.hypotheses = []
                ## To-do: make self.evals into a dict for neater code ##
                self.ents = []
                self.conts = []
                self.neuts = []

                self.turns += 1

            if self.turns == 1:
                # Hypothesis writing
                for agent in self.writers_copy:
                    hypothesis = agent.act(blocking=False, turn_timeout=500)
                    if hypothesis is not None:
                        # self.hypotheses.append(hypothesis)
                        # self.writers_copy.remove(agent)
                        # If worker timed out, let them know.
                        if 'incomplete' in hypothesis:
                            agent.observe(self.incomplete_message)
                            self.hypotheses.append({'id':hypothesis['id'],
                                'text': '<i>No submission. Please mark as invalid and rank second</i>', 
                                'task_data': '<i>No submission. Please mark as invalid and rank second</i>',
                                'task_data2': '<i>No submission. Please mark as invalid and rank second</i>',})
                        else: 
                            self.hypotheses.append(hypothesis)
                        self.writers_copy.remove(agent)
                        if len(self.writers_copy) == 0:
                            self.turns +=1

            if self.turns == 2:
                # Collect all the hypothesis and pass the first set to the rankers
                # Sort out hypotheses: assign hypothesis to correct worker
                self.hypotheses_collect = {}
                for i in range(len(self.hypotheses)):
                    for agent in self.agents:
                        if self.hypotheses[i]['id'] == agent.demo_role:
                            num = int(agent.demo_role[-1])
                            self.hypotheses_collect[num-1] = self.hypotheses[i]

                # Writers observe the other writer's hypotheses
                for i in self.sets.keys():
                    showme = [self.hypotheses_collect[(i*2)+1], self.hypotheses_collect[(i*2)]] # 1,3 / 0,2
                    for j in range(2):
                        self.sets[i][j].observe({'id':'Other writer\'s claims', 'text':'\n<u>Def. correct</u>: ' + showme[j]['text'] + '\n<u>Def. incorrect</u>: ' + showme[j]['task_data'] + '\n<u>Neither</u>: ' + showme[j]['task_data2']}) #1,3
                # Pause to read claims
                time.sleep(10)

                # Sort out hypotheses according to label and writer
                self.entailments, self.contradictions, self.neutrals  = {}, {}, {}
                self.map_entail, self.map_contradict, self.map_neutral = {}, {}, {} # to store flip maps
                for i in range(self.num_agents):
                    self.entailments[i] = {'id':'Claim '+str((i%2)+1), 'text': self.hypotheses_collect[i]['text'], 'task_data': {'respond_with_form':[], 'writing':False, 'onboard':False}}
                    self.contradictions[i] = {'id':'Claim '+str((i%2)+1),'text': self.hypotheses_collect[i]['task_data']}
                    self.neutrals[i] = {'id':'Claim '+str((i%2)+1),'text': self.hypotheses_collect[i]['task_data2']}

                print("NOTEY: hypothesis written")

                # Show agents the prompt from the other set
                # And the claims for that prompt
                for agent in self.agents:
                    agent.observe({'id':'Phase 2', 
                                   'text':'<b>Thank you for writing your claims! Now please read the following prompt and rank the claims written by other people,</b>'})
                time.sleep(3)
                label = 'Definitely correct'
                # Round robin exchange of prompts and claims for ranking
                setnum = itertools.cycle(range(len(self.sets)))
                next(setnum)
                for i in self.sets.keys():
                    num = (i+1) % len(self.sets)
                    prompt = self.prompts[num]
                    self.map_entail[self.sets[i][0]], self.map_entail[self.sets[i][1]] = self.observe_hypotheses(prompt, label, self.sets[i], self.entailments[num*2], self.entailments[(num*2)+1])

                print("NOTEY: ranking started")  
                # evaluation, evaluation2, evaluation3 = None, None, None
                self.turns += 1

            if self.turns == 3:
                # Ranking entailments
                # Show entailment set         
                for agent in self.evaluators_ent:
                    evaluation = agent.act(blocking=False, turn_timeout=300)#300
                    if evaluation is not None:
                        if 'incomplete' in evaluation:
                            agent.observe(self.incomplete_message)
                            self.ents.append({'id':evaluation['id'],
                                             'text': None,
                                             'task_data':'',
                                             'task_data2': None,
                                             'task_data3': None})
                        else:
                            self.ents.append(evaluation)
                        self.evaluators_ent.remove(agent)
                        if agent not in self.evaluators_cont:
                            self.evaluators_cont.append(agent)
                            self.keep_evaluator_status.append(agent)
                            for i in self.sets.keys():
                                if agent in self.sets[i]:
                                    num = (i+1) % len(self.sets)
                                    prompt = self.prompts[num]
                            label = 'Definitely incorrect'
                            show_hypotheses = self.contradictions
                            mappy = self.map_contradict # shallow copy
                            mappy[agent], _ = self.observe_hypotheses(prompt, label, [agent], show_hypotheses[num*2], show_hypotheses[(num*2)+1])

                # Show contradiction set
                for agent in self.evaluators_cont:
                    evaluation2 = agent.act(blocking=False, turn_timeout=300)#300
                    if evaluation2 is not None:
                        if 'incomplete' in evaluation2:
                            agent.observe(self.incomplete_message)
                            self.conts.append({'id':evaluation2['id'],
                                             'text': None,
                                             'task_data':'',
                                             'task_data2': None,
                                             'task_data3': None})
                        else:
                            self.conts.append(evaluation2)
                        self.evaluators_cont.remove(agent)
                        if agent not in self.evaluators_neut:
                            self.evaluators_neut.append(agent)
                            self.keep_evaluator_status.append(agent)
                            for i in self.sets.keys():
                                if agent in self.sets[i]:
                                    num = (i+1) % len(self.sets)
                                    prompt = self.prompts[num]
                            label = 'Neither definitely correct nor definitely incorrect'
                            show_hypotheses = self.neutrals
                            mappy = self.map_neutral # shallow copy
                            mappy[agent], _ = self.observe_hypotheses(prompt, label, [agent], show_hypotheses[num*2], show_hypotheses[(num*2)+1])

                # Show neutral set
                for agent in self.evaluators_neut:
                    evaluation3 = agent.act(blocking=False, turn_timeout=300)#300
                    if evaluation3 is not None:
                        if 'incomplete' in evaluation3:
                            agent.observe(self.incomplete_message)
                            self.neuts.append({'id':evaluation3['id'],
                                             'text': None,
                                             'task_data':'',
                                             'task_data2': None,
                                             'task_data3': None})
                        else:
                            self.neuts.append(evaluation3)
                        self.evaluators_neut.remove(agent)
                        if len(self.evaluators_neut) == 0 and (len(self.keep_evaluator_status) == len(self.agents)*2):
                            print("NOTEY: ranking done")
                            self.turns += 1

            if self.turns == 4:
                # Give feedback to rankers and writers
                # Currently not checking for validaton agreement
                ## To-do: cleanup code in this section. it's a hot mess and might break with > 4 agents ##

                persons = [agent.demo_role for agent in self.agents]
                label_types = ['Definitely Correct', 'Definitely Incorrect', 'Neither']
                self.evals =[self.ents, self.conts, self.neuts]
                self.maps = [self.map_entail, self.map_contradict, self.map_neutral]
                hypothesis_types = [self.entailments, self.contradictions, self.neutrals]
                set_evals = {}
                for i in self.sets.keys():
                    set_evals[i] = []
                    for evals in self.evals:
                        # 0-> 0,1 ; 1 -> 2,3
                        set_eval_ = [next(item for item in evals if item['id'] == persons[i*2]), next(item for item in evals if item['id'] == persons[(i*2)+1])]
                        set_evals[i].append(set_eval_)

                writer_feedback = []
                evaluator_feedback = []
                worker_bonuses = {key: 0.0 for key in self.agents}
                setnum = itertools.cycle(range(len(self.sets)))
                next(setnum)
                for j in self.sets.keys():
                    eval0_justifications = []
                    eval1_justifications = []
                    w0_rank = 0
                    w1_rank = 0
                    agrm_rate = 0
                    evaluators = self.sets[j]
                    # num = next(setnum)
                    num = (j+1) % len(self.sets)
                    writers = self.sets[num]
                    for i, rankings in enumerate(set_evals[j]):
                        label = label_types[i]
                        both_hypotheses = [hypothesis_types[i][num*2]['text'], hypothesis_types[i][num*2 + 1]['text']]
                        # hypothesis0 = self.all_hypotheses[i][j*2]
                        # hypothesis1 = self.all_hypotheses[i][j*2 + 1]
                        eval0_explain = ''
                        eval1_explain = ''

                        writer_bonus_message = {'id': label, 'text': 'You ranked 1st! Bonus = $' + str(self.writer_bonus) +'.'}
                        writer_rank2_message = {'id': label, 'text': 'Unfortunately, you ranked 2nd.'}
                        noagrm_bonus_message = {'id': label, 'text': 'The evaluators did not agree. Bonus = $' + str(self.ranker_bonus) + '.'} # Set as ranker_bonus to avoid incentivizing workers to always disagree on ranking
                        writer_incomplete_message = {'id': label, 'text': 'Unfortunately both evaluators timed out and did not submit rankings.'}
                        
                        agrm_bonus_message = {'id': label, 'text': 'You agreed with the other evaluator! Bonus = $' + str(self.ranker_bonus) + '.'}
                        noagrm_nobonus_message = {'id': label, 'text': 'You and the other ranker disagreed on the ranking.'}
                        eval_you_incomplete = {'id': label, 'text': 'Unfortunately you timed out and did not submit a ranking.'}
                        eval_other_incomplete = {'id': label, 'text': 'Unfortunately the other evaluator timed out and did not submit a ranking.'}

                        # Map the selected choices to "unflip" the ordering
                        # and collect ranker justifications
                        if rankings[1]['id'] == persons[(j*2)+1]: # Even numbered Persons 
                            this_ranker = self.agents[int(persons[(j*2)+1][-1])-1]
                            # rankings[1]['text'] = self.maps[i][this_ranker][rankings[1]['text']]
                            if rankings[0]['task_data'] is not '':
                                eval0_justifications.append(label + ': ' + rankings[0]['task_data'])
                                eval0_explain = rankings[0]['task_data']
                            if rankings[1]['task_data'] is not '':
                                eval1_justifications.append(label + ': ' + rankings[1]['task_data'])
                                eval1_explain = rankings[1]['task_data']
                        else:
                            this_ranker = self.agents[int(persons[j*2][-1])-1]
                            # rankings[0]['text'] = self.maps[i][this_ranker][rankings[0]['text']]
                            if rankings[0]['task_data'] is not '':
                                eval1_justifications.append(label + ': ' + rankings[0]['task_data'])
                                eval1_explain = rankings[0]['task_data']
                            if rankings[1]['task_data'] is not '':
                                eval0_justifications.append(label + ': ' + rankings[1]['task_data'])
                                eval0_explain = rankings[1]['task_data']

                        # Show messages about bonuses
                        if rankings[0]['text'] == rankings[1]['text']:
                            if rankings[0]['text'] is None:
                                evaluator_feedback.append((evaluators[0], eval_you_incomplete))
                                evaluator_feedback.append((evaluators[1], eval_you_incomplete))
                                for writer in writers:
                                    writer_feedback.append((writer, writer_incomplete_message))
                            else:
                                agrm_rate += 1
                                for evaluator in evaluators:
                                    evaluator_feedback.append((evaluator, agrm_bonus_message))
                                    worker_bonuses[evaluator] += self.ranker_bonus
                                if rankings[0]['text'] == 'Claim 1':
                                    writer_feedback.append((writers[0], writer_bonus_message))
                                    worker_bonuses[writers[0]] += self.writer_bonus
                                    writer_feedback.append((writers[1], writer_rank2_message))
                                    w0_rank += 1
                                else:
                                    writer_feedback.append((writers[1], writer_bonus_message))
                                    worker_bonuses[writers[1]] += self.writer_bonus
                                    writer_feedback.append((writers[0], writer_rank2_message))
                                    w1_rank += 1
                        else:
                            # Rankers did not agree
                            ## Tell writers
                            for writer in writers:
                                writer_feedback.append((writer, noagrm_bonus_message))
                                worker_bonuses[writers[1]] += self.ranker_bonus
                                w0_rank += 1
                                w1_rank +=1
                            ## Tell rankers
                            ### Message if timed out
                            if rankings[0]['text'] == None:
                                evaluator_feedback.append((evaluators[0], eval_you_incomplete))
                                evaluator_feedback.append((evaluators[1], eval_other_incomplete))
                            elif rankings[1]['text'] == None:
                                evaluator_feedback.append((evaluators[0], eval_other_incomplete))
                                evaluator_feedback.append((evaluators[1], eval_you_incomplete))
                            else: # Disgree on ranking
                                for evaluator in evaluators:
                                    evaluator_feedback.append((evaluator, noagrm_nobonus_message))

                        if eval0_explain != '':
                            evaluator_feedback.append((evaluators[1], {'id':'Explanation from other evaluator', 'text': eval0_explain}))
                            for writer in writers:
                                writer_feedback.append((writer, {'id':'Evaluator 1\'s explanation', 'text': eval0_explain}))
                        if eval1_explain != '':
                            evaluator_feedback.append((evaluators[0], {'id':'Explanation from other evaluator', 'text': eval1_explain}))
                            for writer in writers:
                                writer_feedback.append((writer, {'id':'Evaluator 2\'s explanation', 'text': eval1_explain}))
                        
                        # Show writers their and the other writer's 
                        # claims again
                        for n, writer in enumerate(writers):
                            writer_feedback.append((writers[n], {'id':'Your claim', 'text':both_hypotheses[n]}))
                            writer_feedback.append((writers[n], {'id':'Other claim', 'text':both_hypotheses[(n+1)%2]}))

                    # Sort out evaluator feedback+bonues by agent and show the messsages
                    evaluator_i_feedback = {}
                    for i, evaluator in enumerate(evaluators):
                        evaluator_i_feedback[i] = [evaluator_feedback[j][1] if evaluator_feedback[j][0] == evaluators[i] else None for j in range(len(evaluator_feedback))]
                    self.give_feedback(evaluator_i_feedback, evaluators)

                # Pause before showing feedback for writing
                time.sleep(5)

                # Show writing feedback and bonuse amounts
                writer_i_feedback = {}
                for i, writer in enumerate(self.agents):
                    writer_i_feedback[i] = [writer_feedback[j][1] if writer_feedback[j][0] == writer else None for j in range(len(writer_feedback))]
                self.give_feedback(writer_i_feedback, self.agents, writing=True)

                time.sleep(5)
                for agent in self.agents:
                    agent.observe({"id":"Your total bonus for this round is", 
                                  "text":"$"+str(worker_bonuses[agent])+" (the bonus is not instantaneous but it will be sent within 24hrs.)"})

                # Organize data for future nikita
                label_types = ['entailment', 'contradiction', 'neutral']
                # data_dump = pd.DataFrame(columns=['label', 'prompt', 'claim-1', 'claim-2', 'claim1-val-p1', 'claim1-val-p2', 'claim2-val-p1', 'claim2-val-p2', 'rank-p1', 'rank-p2', 'expl-p1', 'expl-p2', 'eval-p1', 'eval-p2', 'p1', 'p2', 'workers'])
                data_dump = []
                worker_ids = [(agent.demo_role, agent.worker_id) for agent in self.agents]
                for i, evals in enumerate(self.evals):
                    done = []
                    for j in range(len(evals)):
                        skip = False
                        for num in range(int(self.num_agents/2)):
                            if evals[j]['id'] in  persons[num*2:(num*2)+2]: # 0->1,2 ; 1->3,4
                                if evals[j]['id'] in done:
                                    skip = True
                                else:
                                    setnum = num
                                    done += persons[num*2:(num*2)+2]
                        if not skip:
                            this_data ={}
                            this_data['p1'] = persons[num*2:(num*2)+2][0]
                            this_data['p2'] = persons[num*2:(num*2)+2][0]
                            this_data['label'] = label_types[i]
                            this_data['prompt'] = self.prompts[setnum]['text']
                            this_data['claim-1'] = hypothesis_types[i][setnum*2]['text']
                            this_data['claim-2'] = hypothesis_types[i][(setnum*2)+1]['text']

                            eval_persons = persons[((setnum+1)%2)*2:(((setnum+1)%2)*2)+2]
                            this_data['eval-p1'] = eval_persons[0]
                            this_data['eval-p2'] = eval_persons[1]
                            eval_p1 = list(filter(lambda x: x['id']==eval_persons[0], evals))[0]
                            eval_p2 = list(filter(lambda x: x['id']==eval_persons[1], evals))[0]
                            this_data['claim1-val-p1'] = eval_p1['task_data2']
                            this_data['claim2-val-p1'] = eval_p1['task_data3']
                            this_data['claim1-val-p2'] = eval_p2['task_data2']
                            this_data['claim2-val-p2'] = eval_p2['task_data3']

                            this_data['rank-p1'] = eval_p1['text']
                            this_data['rank-p2'] = eval_p2['text']

                            this_data['expl-p1'] = eval_p1['task_data']
                            this_data['expl-p2'] = eval_p2['task_data']
                            this_data['workers'] = worker_ids

                            data_dump.append(this_data)

                data_dump = pd.DataFrame(data_dump)
                print(data_dump)
                self.interim_data.append(data_dump.to_dict()) # dict so we don't require pickling

                # Pause before moving to next prompt
                time.sleep(10)
                if self.meta_turn+1 < self.max_meta_turns:
                    for agent in self.agents:
                        agent.observe({'id':'<font color="black">Next round</font>', 
                                       'text':'<font color="black"><b>Thank you! In a moment we will start the next round. Once again, you\'ll be asked to write claims for a new prompt.</b></font>'})
                time.sleep(3)

                print("NOTEY: turn completed")
                self.meta_turn += 1
                self.turns = 0

        else:
            print("NOTEY: episode completed")
            for agent in self.agents:
                agent.observe({'id':'<font color="black">Next round</font>', 
                               'text':'<font color="black"><b>Thank you! You have completed this HIT!</b></font>'})
            self.episodeDone = True

    def observe_hypotheses(self, prompt, label, agents, hyp0, hyp1):
        """
        Flip a coin and switch order of claims if flip=1
        Agents observe relevant prompt, label, and hypotheses
        Returns the mapping of flip
        """
        def make_observation(agent, hyp_0, hyp_1):
            if 'task_data' in hyp_0:
                agent.observe({'id':'Prompt', 'text': prompt['text'] + 
                                            '\n<b>Label</b>: '+ label +
                                            '\n\n<b>'+hyp_0['id']+'</b>: '+hyp_0['text'] + 
                                            '\n<b>'+hyp_1['id']+'</b>: '+hyp_1['text'],
                                            'task_data': hyp_0['task_data'] })
            else:
                agent.observe({'id':'Prompt', 'text':prompt['text'] + 
                                            '\n<b>Label</b>: '+ label +
                                            '\n\n<b>'+hyp_0['id']+'</b>: '+hyp_0['text'] + 
                                            '\n<b>'+hyp_1['id']+'</b>: '+hyp_1['text']})

        # To-do: remove unnecessary if continuing to not do a coin flip.
        # flip = random.randint(0,1)
        flip = 0
        mappy = [None, None]
        if flip == 0:
            # mappy_0 = self.noflip
            for i, agent in enumerate(agents):
                make_observation(agent, hyp0, hyp1)
                mappy[i] = self.noflip
        else:
            flip0 = copy.deepcopy(hyp0)
            flip1 = copy.deepcopy(hyp1)
            flip0['id'] = 'Claim 2'
            flip1['id'] = 'Claim 1'
            if len(agents) == 2:
                make_observation(agents[0], hyp0, hyp1)
                make_observation(agents[1], flip1, flip0)
                mappy[0] = self.noflip
                mappy[1] = self.yesflip
            elif len(agents) == 1:
                make_observation(agents[0], flip1, flip0)
                mappy[0] = self.yesflip
            else:
                assert "Do not currently support more than 2 agents in a single set."
        return mappy[0], mappy[1]

    def give_feedback(self, all_feedback, agents, writing=False):
        for i in all_feedback.keys():
            text = ''
            for feedback in all_feedback[i]:
                if feedback is None:
                    pass
                elif 'No bonus' in feedback['id']:
                    text += '\n' + feedback['text']
                elif 'explanation' in feedback['id'].lower():
                        text += '\n<u>' + feedback['id'] + '</u>: ' + feedback['text']
                elif 'claim' in feedback['id']:
                    text += '\n<i><u>' + feedback['id'] + '</u>: ' + feedback['text'] + '</i>'
                else: #label
                    text += '\n\n<b>' + feedback['id'] + '</b>: ' + '\n' + feedback['text']
            if not writing: # evaluation feedback
                agents[i].observe({'id':'Thank you! Here are your bonuses for ranking based on agreement with the other evaluator', 'text':text, 'task_data': {'respond_with_form':[], 'feedback':True}})
            else: # writing feedback
                agents[i].observe({'id':'And these are your bonuses for the claims you wrote in Phase 1', 'text':text, 'task_data': {'respond_with_form':[], 'feedback':True}})

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        # Parallel shutdown of agents
        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  # not MTurkAgent

        threads = []
        for agent in self.mturk_agents:
            t = threading.Thread(target=shutdown_agent, args=(agent,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def review_work(self):
        # Can review the work here to accept or reject it
        for agent in self.agents:
            agent.approve_work()
        pass

    def get_custom_task_data(self):
        # brings important data together for the task, to later be used for
        # creating the dataset. If data requires pickling, put it in a field
        # called 'needs-pickle'.
        return {
            'hit-data': self.interim_data,
        }
