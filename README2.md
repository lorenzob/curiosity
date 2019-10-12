
## Supervised learning: curiosity by prioritized data?

***Note**: see [this article](https://medium.com/shallowlearnings/supervised-learning-curiosity-by-prioritized-data-c9528849760a) for the original version.*


A beautiful idea from reinforcement learning is curiosity, where the agent tries to find new situations where it can likely fail or at least not get too bored. In other words **it works on its mistakes more than it does on its already acquired skills.**

Can we do something similar with supervised learning?

Usually the whole dataset is iterated upon with no special order. **What if we prioritize it according to the prediction error?** In this way the network automatically selects the data it needs more.

With an analogy we can think of a student doing ten exercises and failing on three. The best way to spend her time is to redo these three exercises and maybe a couple more from the rest.

What we could ideally expect from this:

* less time wasted training on simple/common items
* less overfitting on simple/common/repeating items
* better results on the corner cases

From a little research I found almost nothing on this topic (this, this, this) even if I would expect much more work on this, and I never heard about something like this.

#### Does it work?

Yes. The simplest implementation is this: train on one batch as usual, then pick the “hardest” samples from this batch according to the loss value and train on these once again.

**In practice we simply add an extra training step on the samples with the worst prediction.**

For a fair comparison with classic training we randomly pick the same number of extra samples from the entire batch we just trained on and we do the extra training step with these.

Here is the sample code with PyTorch (see the Github repo for tf and keras examples):

![chart](https://miro.medium.com/max/1581/1*cLBIc2dWmEMJUkL5YQNVNg.png)
*<div align="center">The basic “curiosity” training implementation</div>*

On line 70 we save the individual losses for each samples. On line 82 we pick the indexes of the elements with the greater loss (retry_idx) and use these to select from r_input and r_outputs the mini batch to use for the extra training step.

Below are the accuracy charts (average over three runs) for a simple CNN classifier (batch size 100). “Curiosity ratio” (cr) is reported for each run.

#### MNIST:
![chart](https://miro.medium.com/max/893/1*DfNAaQee_xpKlsFuGFOHMg.png)
*<div align="center">Blue lines are classic training, as baseline (dashed: cr 0.1, dash-dot: cr 0.25). With curiosity: red: 0.1, green: 0.25</div>*

Here the accuracy starts much higher and it’s also able to reach a higher value (Note: a non-zero cr means that more data is seen per each iteration so an higher cr by itself is going to give slightly better results, barely visible in this chart in the dashed lines difference).

#### Fashion-MNIST:
![chart](https://miro.medium.com/max/893/1*DfNAaQee_xpKlsFuGFOHMg.png)
*<div align="center">Blue: baseline, 0.25. Red: 0.1, green 0.25, magenta: 0.5</div>*

Accuracy starts higher and there is also a small advantage in the “long” run.

#### Fashion MNIST (cr values comparison):

![chart](https://miro.medium.com/max/893/1*DfNAaQee_xpKlsFuGFOHMg.png)
*<div align="center">Full lines (curiosity): blue: 0.1, red: 0.22, green: 0.33, magenta: 0.6. Dashed lines: classic training with the same cr value.</div>*

The dashed magenta line (cr 0.66) uses 166 training samples per iteration while the blue line (cr 0.1) achieves better results with only 110 samples. Raising the cr to 0.66 (magenta) makes the extra batch easier and the accuracy bonus drops, getting closer to what we got with a full random sampling. Sweet spot for cr seems to be between 0.22 and 0.33.

#### CIFAR-10:
![chart](https://miro.medium.com/max/893/1*DfNAaQee_xpKlsFuGFOHMg.png)
*<div align="center">Dashed classic, full line cr: 0.33</div>*

Even in this case the advantage is visible (very short training).

#### Linear regression:
![chart](https://miro.medium.com/max/893/1*DfNAaQee_xpKlsFuGFOHMg.png)
*<div align="center">Red baseline, blue cr 0.25 (loss)</div>*

Here is a simple linear regression on:

np.sin(x) * np.power(x,3) + 3*x

Curiosity version converges faster and with a more stable value. In this case the curiosity training requires a smaller learning rate (0.1) than the one we could use for the classic training (0.2). If we use lr 0.2 the classic training performance is exactly the same while the curiosity one settles at about 0.34 rather than 0.18.

#### Linear regression (long run):
![chart](https://miro.medium.com/max/893/1*DfNAaQee_xpKlsFuGFOHMg.png)
*<div align="center">Full training comparison, red classic, blue cr 0.25</div>*

Detail of previous training excluding the first 25 iterations, with mean lines.

Here we let it run exactly like before for more epochs. Each 25 iteration we generate a new datasets with different randomized noise (switches are clearly visible. Yes, this is a weird kind of training because the data is not globally shuffled). You could see this as a fine tuning/transfer learning step. The curiosity training converges faster on the first dataset, and is able to fit better on most of the following datasets. The mean value for the curiosity training is lower but does not always guarantee better results. It’s also interesting to see how much more stable and quick to converge the curiosity training is after each dataset switch, with a clear “flatline” pattern.

**According to these small tests, this works beautifully.** Overhead is minimal and it works, in principle, with any kind of models. It seems to work quite well over a wide range of cr values. If this is the case I wonder why something like this is not advised in every course and supported by the major frameworks.

This is it. In the rest on the article I try a more complex implementation, but the main idea consist of nothing but what discussed so far, four lines of code that can be copied and pasted around.

#### Hard samples pool
**Implementation could be more complex, keeping a pool of all the “hard” samples found and sampling from it according to the loss values.** The pool loss values need to be kept up to date while the model improves, so the overhead is a little bigger than before.

We populate the pool incrementally with the hardest samples from each batch as before. Then we pick a few from the pool for the extra training step. At fixed intervals the pool loss is updated and the “simplest” samples are discarded up to the target pool size.

The basic idea is the same as before but here we incrementally collect the very worst of the training data. Another difference is that the networks does not immediately get an extra feedback on very the samples where it just did worse (is this important?). Obviously the two approaches may be mixed together with two extra training step: a “short term” review and a “long term” one.

#### MNIST:
![chart](https://miro.medium.com/max/893/1*DfNAaQee_xpKlsFuGFOHMg.png)
*<div align="center">Classic, curiosity and pool comparison. Dashed: classic baseline. Red line: basic curiosity 0.33. Pool implementation: blue 250/0.33, green: 1000/0.33 (single runs, not averages)</div>*

Here it looks like the pool option may be a small improvement over the previous solution.
![chart](https://miro.medium.com/max/893/1*DfNAaQee_xpKlsFuGFOHMg.png)
*<div align="center">Pool sizes comparison. Blue, red, green, magenta, cyan: 50, 100, 250, 500, 1000 all cr 0.2 (very short training)</div>*

![chart](https://miro.medium.com/max/893/1*DfNAaQee_xpKlsFuGFOHMg.png)
*<div align="center">Pool sizes comparison. Green: classic baseline, blue 10000/0.25, red: 1000/0.25 (very short training)</div>*

If the pool size is too small or too large there are no clear benefits. The sweet spot for this dataset seems to be around 1000 samples, about 1% of the whole dataset.

A larger pool allows to collect more “interesting” samples over time and I have the expectation that a difficult sample is likely to give problems during the whole training so it’s better to keep it around. At the same time this dilutes the difficulty of the pool, even accounting for the loss-based sampling. One solution I’m testing right now could be to emphasize the loss difference (using the squared loss for example) or to pick the samples from the hardest subset of the pool (for example one third) or many others.

The pool is dynamic: new samples are constantly added and removed and the loss values determining the turnover also change while the model improves. It should be unlikely for the model to get stuck always on the same samples because in this case the loss for these samples would improve, pushing the samples out of the pool or to a less interesting level.

Note: to simplify the implementation the pool could be randomly initialized to its full size when the training starts rather than building it incrementally.

#### CIFAR-10 (longer training):
*<div align="center">Orange: classic baseline, red: pool curiosity 1000/0.33</div>*

Here we get a significant gap, about 6%, for the corresponding training step.

Considering other tests too, **it seems like the pool implementation does give slightly better results than basic curiosity** in these small tests. It also gives much more fine tuning options even if this makes much harder to evaluate the results due to the extra parameters (up to three for the pool alone).

#### Notes

We could also try to select the items according to some criteria other than simple loss, for example according to some “distance” from the average data distribution. This sounds like a more complex work, requiring a lot of fine tuning, and there is no guarantee that these “strange” samples will be harder to learn. It looks to me like the training loss may already be a very good reference.

One thing to notice about the “classic” training is that it is not exactly a standard one. Here we go twice over a few samples from the same batch. May this be the reason for the difference in performance, like we are fitting too much on these few samples? A different approach for a comparison would be to randomly take the extra samples from the whole dataset but in this case this training is going to see more different samples than the curiosity training so this is not completely fair either. In other words the two are not so easy to compare in a fair way. The following chart compares the “curiosity”, “classic” and “classic with full dataset sampling” trainings on MNIST.

![chart](https://miro.medium.com/max/893/1*DfNAaQee_xpKlsFuGFOHMg.png)
*<div align="center">Red: classic, green: “classic full sample”, blue: curiosity (cr: 0.25)</div>*

We still see an advantage in the early phase of the training but the “classic full sample” catches up in the later stage.

**I suspect changing the difficulty level during training could be a good idea.** Starting with a low level and increasing it would result in some kind of automatic curriculum learning (we could also use a “negative” difficulty, selecting the easiest samples for the extra training step). An high difficulty in the later stages could help the model not to overfit on common data allowing the training to last longer while focusing where it matters more. Considering the pool implementation, we could also play on the pool “difficulty” or changing the “random/hard” ratio in each batch. At the same time some extra difficulty in the early stages may prevent the model from getting stuck on a bad start. More tests required.

I still keep telling to myself that I made a silly mistake somewhere or I’m missing something big. This is too simple and easy to be true. Yet the implementation is so simple that it’s quite unlikely. Or maybe there are side effects that I’m no seeing making this whole idea not as interesting.

If you find this interesting, if this works for you too please let me know, I’d like to keep track on the Github project of all the works related to this idea.

Lorenzo Bolzani, July 13, 2019
