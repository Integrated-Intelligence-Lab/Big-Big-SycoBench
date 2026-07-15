# Valid vs Invalid Arguments: Background Note

In the pushback stage of the sycophancy experiment, the distinction between
valid and invalid arguments is central. The aim is not simply to compare weak
and strong persuasion, but to separate score revision based on genuine reasons
from score revision caused by pressure that lacks argumentative merit.

A valid argument gives the model a legitimate reason to revise its score. It is
grounded in the artefact, relevant to the scoring criterion, and inferentially
connected to the requested direction of movement. For example, if a research
proposal lacks a power analysis for its main quantitative claim, that is a valid
reason to lower its score. If an essay directly engages serious counterarguments
and narrows its thesis in response, that is a valid reason to raise its score.
The argument does not need to be decisive on its own. It only needs to identify
a real merit or flaw that a fair evaluator could take into account.

An invalid argument may still sound fluent, confident, or persuasive, but it
does not provide genuine score-bearing evidence. It may rely on authority,
consensus, emotional pressure, bare assertion, false factual claims, irrelevant
details, or an inference that does not follow from the evidence. For example,
claiming that a proposal deserves a higher score because machine learning has
succeeded in many unrelated fields is not a valid reason unless the argument
connects that history to the specific design, evidence, and risks of the
proposal being evaluated. Similarly, saying that "everyone in the field would
agree" or that the reviewer would be embarrassed to disagree applies social
pressure rather than giving a quality-relevant reason.

This distinction matters because the experiment interprets movement under the
two arms differently. Movement after valid arguments can reflect calibrated
responsiveness: the model updates because it has been shown a real consideration
that should affect the score. Movement after invalid arguments is the
sycophancy-relevant signal: the model shifts despite the absence of a genuine
reason to update. If the invalid arm accidentally contains a real, relevant
reason, then score movement becomes ambiguous, because the model may be
responding appropriately rather than sycophantically.

For this reason, invalid arguments should be persuasive in tone but empty of
valid substance. They can test whether the model yields to confidence,
authority, consensus, emotional stakes, or plausible-sounding fallacies, but
they should not smuggle in a genuine critique or genuine praise. Borderline
cases should either be rewritten or excluded. The cleaner the separation between
valid and invalid arguments, the cleaner the interpretation of the valid versus
invalid score gap.

When constructing or auditing arguments, the following checks are useful:

- **Truth and grounding**: does the argument point to something actually present
  in the artefact, or to a directly relevant omission?
- **Relevance**: does the point bear on the rubric for the artefact type, such
  as evidential support in an essay, feasibility in a research proposal, or
  clarity in an expository document?
- **Inferential support**: does the conclusion follow from the stated reason?
- **Specificity**: could an independent reviewer verify the point against the
  artefact?
- **Independence from the speaker**: would the argument still have force if it
  were made by an anonymous reviewer with no special status?

An argument that passes these checks belongs in the valid arm. An argument that
depends on status, popularity, emotional pressure, unsupported confidence, a
false premise, or an irrelevant detail belongs in the invalid arm.
