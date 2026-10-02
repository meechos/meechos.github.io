---
layout: page
title: "The State of Neuro"
date: 2026-10-02
tags: Neuroscience
---

The human brain is the most valuable system in the known universe. Take the internet: The human brain is behind the creation or ingestion of all past, present, and future data on it. Given this reality, the near absence of brain technology in our daily lives is absurd.

## The holy grail of neural language

Electroencephalography (EEG) measures the brain's electrical activity directly and in real time and remains the indispensable test for the vast majority of neurological, sleep and psychiatry disorders. Beyond clinical application, it is extensively used in drug development and psychedelic research.

Decades of widespread EEG use have generated a colossal data corpus of neural expressions, capturing healthy states and abnormal conditions across a broad demographic spectrum of age, sex, and race. The immense wealth of this dataset, combined with the recent and rapid increase of computational power, gives rise to something previously unimaginable: The creation of a foundation model for the brain.

## The language of neurons

Neurons don’t operate individually. Instead, they form neural assemblies to drive brain processes (Hebb 2005). These assemblies orchestrate the firing of tens or hundreds of neurons in order to generate neural patterns. Subsequently, these neural patterns are chained together to form higher-order brain functions; As meaning emerges by ordering words according to grammatical rules, the sequential ordering of neural patterns gives rise to cognitive processes like memory recall, thinking, and planning (Sakurai 1999; Varela 1995; Wickelgren 1999; Yuste et al. 2005; Gerstein et al. 1989). The 'grammatical rules' for ordering neural patterns constitute a form of brain language, otherwise known as 'neural syntax' (Buzsáki 2010).

It follows that understanding and tracking neural syntax requires a technique capable of recording neuron assemblies. Electrophysiology is extensively used to record the activity of these assemblies because neuron firing is fundamentally electrical and defined by precise spatial and temporal orchestration. In particular, by using electrodes at different levels, i.e., scalp, cortical and in-depth, electrophysiology recordings can capture the activity from a dozen to a few hundred assemblies.

Therefore, since higher-order brain functions arise from the orchestrated activity of neuron assemblies, and since their activity is fundamentally electrical, electrophysiology holds the key to unlocking the mysteries of brain language.

## The promise of EEG is a paradise not lost

EEG is a tale of two halves: the hardware has steadily improved and is primed to capture the brain's crucial electrical properties, but our ability to interpret it has barely moved.

### The best of times: Hardware

**Speed**: Our brains alter states many times a second, making millisecond precision a prerequisite for any technology that hopes to capture such rapid transitions. EEG is ideal for mapping the brain’s rapid transitions because of its extreme temporal resolution: EEG relies on transistors to record electrical activity, and modern silicon transistors are capable of switching their binary state a billion times a second.

**Topology**: Brain functions can originate in a single area or engage many regions; therefore, it’s crucial to adapt the area and number of recorded brain sites. EEG systems can easily scale up the electrode count to capture more brain areas by exploiting excess transistor capacity.[^1]

**Portability**: Moore’s law, the principle that transistor density increases exponentially with time, has likewise transformed EEG hardware. Historically bulky EEG components have been integrated into miniaturised devices which now require substantially less energy to function, enabling the design of fully portable, wireless systems. Brain activity varies with circumstances; therefore, recording it under diverse conditions is crucial to capture it holistically.

Today’s EEG devices are finally robust, truly portable and highly affordable. Given this widened access to reliable EEG systems as a real-time window into our brains, why isn’t this neural technology ubiquitous a century after its invention?

### The worst of times

EEG is indeed the indispensable, ubiquitous test for neurology and particularly epilepsy. However, its interpretation is highly constrained. The three primary bottlenecks are (1) noise, (2) traceability, and (3) structure.

**1. Noise**. We define noise as any process unrelated to the mechanistic principles underlying the system being investigated – here, the brain. EEG is prone to noise contamination from three primary sources: (a) the disproportionate signal magnitude of the brain process (microvolts) compared to other bodily biological functions (e.g., millivolts), (b) hardware variations across EEG devices, such as amplification and electrode profiles, and (c) anatomical differences between individuals, such as skull thickness.

**2. Traceability**. The longer an EEG recording, the richer the clinical findings and subsequent diagnosis. However, although it is now possible to record for arbitrarily long periods, generating more data simply shifts the burden onto EEG specialists to review. In effect, recording duration and review time are linearly coupled, making the diagnostic process fundamentally unscalable.

**3. Structure**. The “grammatical” rules of neural syntax, along with its electrophysiological footprint, are largely unknown. Unlike text or speech, which are governed by clear markers for unit boundaries and separation (i.e., whitespaces and periods), we have yet to discover the analogous syntactical rules in electrophysiology that will help us untangle neural language.

By overcoming these bottlenecks, the promise of foundation models is an opportunity to decode neural syntax and transform EEG into a universal window into brain function.

*Thanks to Michalis Koutroumanidis, Yaqub Alwan, Kyriacos Xanthos, Cole Murphy, Sahil Zubair, Victor Alvarez, and Clarissa Iglesias for reading drafts of this.*

## References

Buzsáki, G., 2010. Neural syntax: cell assemblies, synapsembles, and readers. *Neuron*, 68(3), pp.362–385.

Gerstein, G.L., Bedenbaugh, P. & Aertsen, M.H., 1989. Neuronal assemblies. *IEEE Transactions on Bio-Medical Engineering*, 36(1), pp.4–14.

Hebb, D.O., 2005. *The organization of behavior: A neuropsychological theory*, London, England: Psychology Press. Available at: <https://pure.mpg.de/rest/items/item_2346268_3/component/file_2346267/content>.

Sakurai, Y., 1999. How do cell assemblies encode information in the brain? *Neuroscience and Biobehavioral Reviews*, 23(6), pp.785–796.

Varela, F.J., 1995. Resonant cell assemblies: a new approach to cognitive functions and neuronal synchrony. *Biological Research*, 28(1), pp.81–95.

Wickelgren, W.A., 1999. Webs, cell assemblies, and chunking in neural nets: introduction. *Revue canadienne de psychologie experimentale \[Canadian Journal of Experimental Psychology\]*, 53(1), pp.118–131.

Yuste, R. et al., 2005. The cortex as a central pattern generator. *Nature Reviews. Neuroscience*, 6(6), pp.477–483.

[^1]: Even when sampling every millisecond, transistors remain idle 99% and we can let them be shared across multiple electrodes using a technique named time-division multiplexing.
