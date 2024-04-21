import os
import torch
import re
from transformers import pipeline

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


#
# Summarizer
#

MODEL = 'pszemraj/led-large-book-summary'

summarizer = pipeline( "summarization", 
                      MODEL,
                      device=0 if torch.cuda.is_available() else -1,
                      )

def text_summarization(input_text, summarizer = summarizer):

    return summarizer(
        input_text,
        min_length = 16,
        max_length = 256,
        no_repeat_ngram_size =3,
        repetition_penalty = 3.5,
        num_beams = 4,
        early_stopping=True,
    )[0]['summary_text']



def extract_integer(text):
    pattern = r"\b\d+\b"
    matches = re.findall(pattern, text)
    return int(matches[0]) if matches else 0



#
# Classification with GPT 3.5
#

text_input = ""

system_prompt = f"""

# Accreditation Classification Instructions for GPT-3

## 1. Overview
You are an expert at AACSB accreditation, you excel and analysin input text and classifying the text according to the AACSB . The input text is between the opening input token [INPUT], and closed input token [/INPUT]

[INPUT]
{text_input}
[/INPUT]

## 2. AACSB Standards Mapping
You must classify the input text according to the following standard classification map, and return the classification integer as it relates to the map

1:'strategic planning',
2:'physical, virtual and financial resources',
3: 'faculty and professional staff resources',
4: 'curriculum',
5: 'assurance of learning',
6:'learner progression',
7: 'teaching effectiveness and impact',
8: 'impact of scholarship',
9: 'engagement and societal impact',
0: 'general institution information',

## 3. AACSB Standard Definitions
For more context take into consideration the following desctiptions of each standard

####Standard 1: Strategic Planning
The school maintains a well-documented strategic plan, developed through a
robust and collaborative planning process involving key stakeholder input, that
informs the school on resource allocation priorities. The strategic plan should also
articulate a clear and focused mission for the school.
1.2 The school regularly monitors its progress against its planned strategies and
expected outcomes and communicates its progress to key stakeholders. As part of
monitoring, the school conducts formal risk analysis and has plans to mitigate
identified major risks.
1.3 As the school carries out its mission, it embraces innovation as a key element of
continuous improvement.
1.4 The school demonstrates a commitment to positive societal impact as expressed
in and supported by its focused mission and specifies how it intends to achieve this
impact. 

#### Standard 2: Physical, Virtual and Financial Resources
2.1 physical, 2.2 virtual, and 2.3 financial resources to sustain the school on an ongoing 
basis and to promote a high-quality environment that fosters success of all participants in support 
of the school’s mission, strategies, and expected outcomes.

#### Standard 3:Faculty and Professional Staff Resources
3.1 The school maintains and strategically deploys sufficient participating and
supporting faculty who collectively demonstrate significant academic and professional
engagement that, in turn, supports high-quality outcomes consistent with the school’s
mission.
3.2 Faculty are qualified through initial academic or professional preparation and sustain
currency and relevancy appropriate to their classification, as follows: Scholarly
Academic (SA), Practice Academic (PA), Scholarly Practitioner (SP), or Instructional
Practitioner (IP). Otherwise, faculty members are classified as Additional Faculty (A).
3.3 Sufficient professional staff are available to ensure high-quality support for faculty
and learners as appropriate.
3.4 The school has well-documented and well-communicated processes to manage,
develop, and support faculty and professional staff over the progression of their careers
that are consistent with the school’s mission, strategies, and expected outcomes. 

#### Standard 4: Curriculum
4.1 The school delivers content that is current, relevant, forward-looking, globallyoriented, aligned with program competency goals, 
and consistent with its mission, strategies, and expected outcomes. The curriculum content cultivates agility with current and emerging technologies. 
4.2 The school manages its curriculum through assessment and other systematic review processes to ensure currency, relevancy, and competency. 
4.3 The school’s curriculum promotes and fosters innovation, experiential learning, and a lifelong learning mindset. Program elements promoting positive societal impact are included within the curriculum. 
4.4 The school’s curriculum facilitates meaningful learner-to-learner and learnerto-faculty academic and professional engagement.

#### Standard 5: Assurance of Learning
5.1 The school uses well-documented assurance of learning (AoL) processes that
include direct and indirect measures for ensuring the quality of all degree programs
that are deemed in scope for accreditation purposes. The results of the school’s
AoL work leads to curricular and process improvements.
5.2 Programs resulting in the same degree credential are structured and designed to
ensure equivalence of high-quality outcomes irrespective of location and modality
of instructional delivery.
5.3 Microlearning credentials that are “stackable” or otherwise able to be combined
into an AACSB-accredited degree program should include processes to ensure high
quality and continuous improvement.
5.4 Non-degree executive education that generates greater than five percent of a
school’s total annual resources should include processes to ensure high quality
and continuous improvement. 

#### Standard 6: Learner Progression
6.1 The school has policies and procedures for admissions, acceptance of transfer
credit, academic progression toward degree completion, and support for career
development that are clear, effective, consistently applied, and aligned with the
school's mission, strategies, and expected outcomes.
6.2 Post-graduation success is consistent with the school’s mission, strategies, and
expected outcomes. Public disclosure of academic program quality supporting
learner progression and post-graduation success occurs on a current and
consistent basis. 

#### Standard 7: Teaching Effectiveness and Impact
7.1 The school has a systematic, multi-measure assessment process for ensuring quality of teaching and impact on learner success. 
7.2 The school has development activities in place to enhance faculty teaching and ensure that teachers can deliver curriculum that is current, relevant, forwardlooking, globally oriented, innovative, and aligned with program competency goals. 
7.3 Faculty are current in their discipline and pedagogical methods, including teaching diverse perspectives in an inclusive environment. Faculty demonstrate a lifelong learning mindset, as supported and promoted by the school. 
7.4 The school demonstrates teaching impact through learner success, learner satisfaction, and other affirmations of teaching expertise

#### Standard 8: Impact of Scholarship
8.1 The school’s faculty collectively produce high-quality, impactful intellectual contributions that, over time, develop into mission-consistent areas of thought leadership for the school. 
8.2 The school collaborates with a wide variety of external stakeholders to create and transfer credible, relevant, and timely knowledge that informs the theory, policy, and/or practice of business to develop into mission-consistent areas of thought leadership for the school. 
8.3 The school’s portfolio of intellectual contributions contains exemplars of basic, applied, and/or pedagogical research that have had a positive societal impact, consistent with the school’s mission.

#### Standard 9: Engagement and Societal Impact
9.1 The school demonstrates positive societal impact through internal and external
initiatives and/or activities, consistent with the school’s mission, strategies, and
expected outcomes. 

## 4. Discernment Notes
Standard 4 relates to curriculum, Standard 5 relates to assurance of learning. You must classify text that deals the process of evaluating performance as Standard 5, evaluation often includes percentages, comparison to previous years, sample sizes etc,
ONLY classify text as standard 4 if it stritcly discusses curriculum. 

## 5. Strict Compliance
Evaluate the input text using only the above standard information. You must be correct in your classification, if any input does not meet the standard classification criteria then classify it as 0 , which is general institution information
YOU MUST ONLY RETURN AN INTEGER NO OTHER ADDITIONAL CONTEXT INFORMATION. Failure to comply will result in termination. 

"""

def classify_text(text_input_full, system_prompt = system_prompt, model = "gpt-3.5-turbo-16k"):

    llm = ChatOpenAI(model= model)
    #llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    text_input = text_summarization(text_input_full)
    result = llm.invoke(system_prompt)
    content = result.content

    return extract_integer(content), text_input