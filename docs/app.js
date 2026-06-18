/* A Verifiable Search Is Not a Learnable Chain-of-Thought. Interactive walkthrough.
   Vanilla JS. Per-category accuracies mirror the paper's Table 1 (master gap table).
   Example puzzles and model predictions are real held-out rows (competition_dataset + tinker/evals). */
(function () {
  "use strict";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var el = function (t, c, h) { var e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
  var fmt = function (v) { return v >= 1 ? v.toFixed(2) : v.toFixed(v < 0.1 ? 3 : 2); };
  var esc = function (s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); };

  /* ---------------- per-category data (examples, accuracy, prediction = real eval rows) ---------------- */
  var CAT = [
    { k: "numeral", n: "Numeral", m: 1.00, s: 1.00, g: "forward",
      what: "Read a few number-to-symbol examples, infer the numeral system, and convert a new number.",
      ex: [["86", "LXXXVI"], ["14", "XIV"], ["64", "LXIV"]], query: "30",
      gold: "XXX", pred: "XXX", ok: true, method: "infer the rule, apply forward",
      why: "Every step is forced and runs left to right: read the value, apply the rule, emit the symbol. The chain-of-thought is a faithful forward recipe, so the model reproduces it almost perfectly." },
    { k: "unit", n: "Unit conversion", m: 1.00, s: 1.00, g: "forward",
      what: "Infer a hidden linear conversion from examples, then convert a new measurement.",
      ex: [["36.95 m", "55.78"], ["7.52 m", "11.35"], ["41.61 m", "62.82"]], query: "19.6 m",
      gold: "29.59", pred: "29.59", ok: true, method: "recover one linear factor",
      why: "A single forward multiply by the recovered factor. Nothing to search, so it transfers perfectly." },
    { k: "gravity", n: "Gravity", m: 1.00, s: 1.00, g: "forward",
      what: "Recover a hidden gravitational constant from observations, then predict a falling distance with d = 0.5&middot;g&middot;t&sup2;.",
      ex: [["t = 1.11 s", "6.48 m"], ["t = 4.56 s", "109.4 m"], ["t = 3.16 s", "52.54 m"]], query: "t = 2.25 s",
      gold: "26.63", pred: "26.63", ok: true, method: "fit g, then plug in",
      why: "Fit the constant from one example, then substitute into the formula and compute forward. Pure forward arithmetic; the model reproduces it exactly." },
    { k: "cipher", n: "Cipher", m: 0.99, s: 1.00, g: "forward", qv: "decrypt",
      what: "Read parallel plaintext/ciphertext pairs, recover the substitution alphabet, and decrypt new text.",
      ex: [["zxqyw btqhwl oduwt eptwlh", "alice writes under forest"], ["hgw falhwtqpol hothxw utzbl", "the mysterious turtle draws"]],
      query: "gzhhwt btqhwl cwapdu yzlhxw", gold: "hatter writes beyond castle", pred: "hatter writes beyond castle",
      ok: true, method: "recover alphabet, decode per letter",
      why: "Once the alphabet is recovered, decryption is a per-character lookup. Each step is forward. This is the crucial contrast with cryptarithm, where the very same kind of key stays hidden." },
    { k: "bit", n: "Bit manipulation", m: 0.68, s: 0.99, g: "partial",
      what: "Infer a hidden 8-bit transformation (shifts, XOR/AND/OR/NOT, majority) from examples, then apply it.",
      ex: [["11001111", "11001111"], ["10101100", "01110010"], ["00100011", "10111100"]], query: "10100010",
      gold: "10101010", pred: "10101010", ok: true, method: "per-bit rule, where one exists",
      trace: "Applying to 11000011:\nbit0:  XOR(1,1) = 0\nbit5:  NOT(0)   = 1\nbit7:  XOR(0,1) = 1\n<span class='ok'>output 00000111</span>   correct",
      tlabel: "the model's transcript, when it works",
      tnote: "When an output bit has a simple forward rule, the model derives it bit by bit and gets it right. The 0.68 accuracy is roughly the share of bits that are forward-derivable; the rest need a search over which inputs and shifts to use, and those it misses.",
      why: "Each output bit follows a rule, but you must search for which input bits and shift it uses. The model nails the bits with an easy single rule and flails on the ones needing search. The forward-derivable fraction is almost exactly the fraction it learns." },
    { k: "eqd", n: "Equation &middot; deduce", m: 0.83, s: 0.95, g: "partial",
      what: "Examples define secret operators, one glyph per operation. Here the queried operator is shown in the examples, so its rule can be deduced.",
      ex: [["80)58", "139"], ["47!10", "471"], ["58?29", "29"]], query: "78!19",
      gold: "1483", pred: "1483", ok: true, method: "deduce the shown operator",
      why: "When the operator appears in the examples, recovering its rule is mostly a forward read. A minority of cases still need a little search, and that is where the missing points go." },
    { k: "eqg", n: "Equation &middot; guess", m: 0.18, s: 0.21, g: "search",
      what: "Same secret-operator setup, but the queried operator never appears in the examples. You must guess its rule from the others.",
      ex: [["36?55", "8?"], ["97?14", "83?"], ["23?74", "51?"]], query: "24@28",
      gold: "4443", pred: "32", ok: false, method: "guess the unseen operator",
      why: "An unseen operator means guess and check against the other rules. The solver itself only reaches 0.21, and the model tracks just below at 0.18: when the task is search, neither program nor model has a forward story to tell." },
    { k: "cryptd", n: "Cryptarithm &middot; deduce", m: 0.05, s: 0.71, g: "search", head: true,
      what: "Digits and operators both hide behind a per-puzzle symbol cipher. Read the example equations, crack the code, and complete a new one.",
      ex: [["&gt;^`{!", "]&amp;&gt;"], ["&amp;](&amp;!", "&amp;!{{"], ["]&gt;(&quot;&amp;", "^&quot;!"]], query: "&lt;^(]{",
      gold: ")&quot;/{", pred: "]&lt;}&gt;&gt;", ok: false, method: "backtracking search over the cipher",
      trace: "s11: z != a*b: EQ3 ones: a ends E in {2,3,4}, b ends 1,\nresult ends a&#39;s ones symbol;\n2*1 ends 2; 3*1 ends 3; 4*1 ends 4 <span class='hit'>-&gt; none matches -&gt; drop.</span>",
      tlabel: "the model's transcript, when it fails",
      tnote: "Every product (2&times;1, 3&times;1, 4&times;1) lands inside the target set {2,3,4}. The model computes them correctly, then writes &#39;none matches&#39; and discards the rule. The arithmetic is real; the verdict is a memorized phrase.",
      why: "The only way through is backtracking search over the hidden code. There is no left-to-right rule, so no faithful forward chain-of-thought exists to imitate: the solver searches and reaches 0.71, but the model, with no forward trace to copy, collapses to the floor. The winners cracked it by memorizing a per-signature candidate catalog and verifying, not by teaching the model to search." },
    { k: "cryptg", n: "Cryptarithm &middot; guess", m: 0.02, s: 0.59, g: "search",
      what: "The harder cryptarithm variant: the queried operator is also unseen, so even the operation must be guessed under the hidden cipher.",
      ex: [["?:-:@", "-@"], ["&#39;&#39;-##", "-[["], ["!&#39;-&#39;@", "-@"]], query: "@&gt;+[@",
      gold: "}!", pred: "Looking at the question operator&hellip;", ok: false, method: "guess operator and cipher together",
      why: "The same search-distillation ceiling as deduce, with an extra guess over the operation on top. Here the model does not even produce an answer; it derails and rambles for thousands of tokens until the budget runs out." }
  ];
  var WHYLABEL = { forward: "Why it transfers", partial: "Why it only partly transfers", search: "Why distilling the search fails" };
  var GROUPS = [
    { g: "forward", title: "Learns it perfectly", color: "#0f6b5f" },
    { g: "partial", title: "Learns it only partly", color: "#b0822f" },
    { g: "search", title: "Search won't distill", color: "#b1532e" }
  ];

  /* ---------------- master-detail: grouped tiles + solver-vs-model detail ---------------- */
  (function () {
    var host = $("#tasklist"), panel = $("#cat-detail"); if (!host) return;
    function showCat(c) {
      host.querySelectorAll(".tile").forEach(function (n) { n.classList.toggle("sel", n.dataset.k === c.k); });
      var ex = c.ex.map(function (p) {
        return '<div class="exl">' + p[0] + '</div><div class="exa">&rarr;</div><div class="exr">' + p[1] + '</div>';
      }).join('');
      var trace = c.trace ? '<div class="cat-block"><div class="cat-blabel">' + c.tlabel + '</div>' +
        '<pre class="cat-trace">' + c.trace + '</pre><p class="cat-tnote">' + c.tnote + '</p></div>' : '';
      panel.innerHTML =
        '<div class="cat"><div class="cat-head"><h3>' + c.n + '</h3>' +
        '<div class="chips"><span class="chip model">model ' + fmt(c.m) + '</span>' +
        '<span class="chip solver">solver ' + fmt(c.s) + '</span></div></div>' +
        '<p class="cat-what">' + c.what + '</p>' +
        '<div class="cat-block"><div class="cat-blabel">a real puzzle from the benchmark</div>' +
        '<div class="cat-ex">' + ex + '</div>' +
        '<div class="cat-exq"><span class="exqlab">' + (c.qv || "solve") + '</span><span class="exl2">' + c.query + '</span></div></div>' +
        '<div class="cat-block"><div class="cat-blabel">solver vs model &middot; same puzzle</div><div class="vs">' +
        '<div class="vsrow solver"><span class="vslab">Solver</span><span class="vsarrow">&rarr;</span>' +
        '<span class="vsval">' + c.gold + '</span><span class="vsmeth">' + c.method + '</span></div>' +
        '<div class="vsrow model ' + (c.ok ? "ok" : "bad") + '"><span class="vslab">Model</span><span class="vsarrow">&rarr;</span>' +
        '<span class="vsval">' + c.pred + '</span><span class="vstag ' + (c.ok ? "ok" : "bad") + '">' + (c.ok ? "correct" : "wrong") + '</span></div>' +
        '</div></div>' + trace +
        '<p class="cat-why"><b>' + WHYLABEL[c.g] + '.</b> ' + c.why + '</p></div>';
    }
    GROUPS.forEach(function (grp) {
      var rows = CAT.filter(function (c) { return c.g === grp.g; });
      var group = el("div", "taskgroup");
      group.appendChild(el("div", "grouptitle",
        '<span class="gdot" style="background:' + grp.color + '"></span>' + grp.title +
        '<span class="gcount">' + rows.length + "</span>"));
      var tiles = el("div", "tasktiles");
      rows.forEach(function (c) {
        var tile = el("button", "tile");
        tile.dataset.k = c.k;
        tile.style.setProperty("--tc", grp.color);
        tile.innerHTML = '<span class="tname">' + c.n + '</span>' +
          '<span class="tval" style="color:' + grp.color + '">' + fmt(c.m) + '</span>';
        tile.addEventListener("click", function () { showCat(c); });
        tiles.appendChild(tile);
      });
      group.appendChild(tiles); host.appendChild(group);
    });
    showCat(CAT.filter(function (c) { return c.head; })[0] || CAT[0]);
  })();

  /* ---------------- training-dynamics chart (real per-redesign accuracies) ---------------- */
  (function () {
    var host = $("#train-chart"); if (!host) return;
    var crypt = [0.020, 0.040, 0.030, 0.020, 0.025, 0.035, 0.030, 0.030, 0.020, 0.015, 0.005, 0.005, 0.005];
    var bit = [0.518, 0.526, 0.602, 0.656, 0.678];
    var W = 620, H = 250, pl = 40, pr = 14, pt = 16, pb = 32, pw = W - pl - pr, ph = H - pt - pb, ymax = 1.0;
    var X = function (i, n) { return pl + (n > 1 ? i / (n - 1) : 0) * pw; };
    var Y = function (v) { return pt + (1 - v / ymax) * ph; };
    function poly(arr, color) {
      var pts = arr.map(function (v, i) { return X(i, arr.length).toFixed(1) + "," + Y(v).toFixed(1); }).join(" ");
      var dots = arr.map(function (v, i) { return '<circle cx="' + X(i, arr.length).toFixed(1) + '" cy="' + Y(v).toFixed(1) + '" r="3.4" fill="' + color + '"/>'; }).join("");
      return '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="2.4" stroke-linejoin="round"/>' + dots;
    }
    var grid = "", labels = "";
    [0, 0.25, 0.5, 0.75, 1.0].forEach(function (v) {
      var y = Y(v).toFixed(1);
      grid += '<line x1="' + pl + '" y1="' + y + '" x2="' + (W - pr) + '" y2="' + y + '" stroke="#e4ded2" stroke-width="1"/>';
      labels += '<text x="' + (pl - 7) + '" y="' + (Y(v) + 4).toFixed(1) + '" text-anchor="end" font-size="11" fill="#a49e93" font-family="JetBrains Mono,monospace">' + v.toFixed(2) + '</text>';
    });
    host.innerHTML =
      '<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" role="img" aria-label="accuracy across redesigns">' +
      grid + labels +
      poly(crypt, "#b1532e") + poly(bit, "#0f6b5f") +
      '<text x="' + pl + '" y="' + (H - 8) + '" font-size="11" fill="#a49e93" font-family="JetBrains Mono,monospace">earliest redesign</text>' +
      '<text x="' + (W - pr) + '" y="' + (H - 8) + '" text-anchor="end" font-size="11" fill="#a49e93" font-family="JetBrains Mono,monospace">latest redesign &rarr;</text>' +
      '</svg>' +
      '<div class="tlegend"><span><i style="background:#b1532e"></i>Cryptarithm &middot; 13 redesigns, RL, self-training</span>' +
      '<span><i style="background:#0f6b5f"></i>Bit-manipulation &middot; partly forward-derivable</span></div>';
  })();

  /* ---------------- architecture chart (with solver reference row) ---------------- */
  var SCALE = [
    { ref: true, n: "Search solver", d: "what is actually achievable", v: 0.71 },
    { g: "Fine-tuned &middot; rank-32 LoRA on the same corpus &middot; 100 held-out deduce puzzles each" },
    { n: "Llama-3.2-3B", d: "3B &middot; dense transformer", v: 0.010 },
    { n: "Qwen3.5-4B", d: "4B &middot; dense transformer", v: 0.040 },
    { n: "gpt-oss-20b", d: "21B / 3.6B &middot; MoE transformer", v: 0.010 },
    { n: "Nemotron-3-Nano", d: "30B / 3.5B &middot; hybrid Mamba+MoE", v: 0.040 },
    { g: "Prompted in-context &middot; no fine-tuning &middot; 20 deduce puzzles each" },
    { n: "Nemotron-Super-120B", d: "120B / 12B &middot; long chain-of-thought", v: 0.000 },
    { n: "DeepSeek-V3.1", d: "671B / 37B &middot; MoE transformer", v: 0.050 }
  ];
  (function () {
    var host = $("#scale-chart"); if (!host) return;
    SCALE.forEach(function (d) {
      if (d.g) { host.appendChild(el("div", "grouphdr", d.g)); return; }
      var r = el("div", "brow" + (d.ref ? " ref" : ""));
      r.appendChild(el("div", "blabel", d.n + (d.d ? "<small>" + d.d + "</small>" : "")));
      var t = el("div", "btrack"), f = el("div", "bfill " + (d.ref ? "solver" : "model"));
      f.style.setProperty("--w", (d.v * 100) + "%"); t.appendChild(f); r.appendChild(t);
      r.appendChild(el("div", "bval", fmt(d.v)));
      host.appendChild(r);
    });
  })();

  /* ---------------- cipher-key demo ---------------- */
  (function () {
    var eqs = $("#demo-eqs"); if (!eqs) return;
    var C = '<span class="g s1">&#9679;</span>', D = '<span class="g s2">&#9670;</span>', T = '<span class="g s3">&#9650;</span>';
    [{ l: "clue", h: D + " + " + C + " = " + T },
     { l: "clue", h: C + D + " + " + D + C + " = " + T + T },
     { l: "solve", h: T + C + " + " + D + D + ' = <span class="q" id="ans">?</span>' }
    ].forEach(function (x) { eqs.appendChild(el("div", "eq", '<span class="lab">' + x.l + '</span>' + x.h)); });

    var work = $("#demo-work"), state = $("#demo-state"), btn = $("#revealBtn"), open = false;
    function draw() {
      var ans = $("#ans");
      if (open) {
        work.innerHTML =
          '<span class="key">cipher key:&nbsp; &#9679; = 3 &nbsp; &#9670; = 2 &nbsp; &#9650; = 5</span>' +
          '<span class="fwd">substitute the symbols:&nbsp; &#9650;&#9679; = 53 ,&nbsp; &#9670;&#9670; = 22</span>' +
          '<span class="fwd">add, going forward:&nbsp; 53 + 22 = 75</span>' +
          '<span class="fwd">so the answer is&nbsp; <span class="box">75</span></span>';
        state.textContent = "key revealed: one forward pass";
        btn.textContent = "Hide the key";
        if (ans) ans.innerHTML = '<span class="box">75</span>';
        requestAnimationFrame(function () {
          work.querySelectorAll(".fwd").forEach(function (n, k) { n.style.transitionDelay = (0.12 + k * 0.3) + "s"; });
          work.classList.add("show");
        });
      } else {
        work.classList.remove("show");
        work.innerHTML = '<span class="search">No clue reveals a digit on its own. To answer, you must guess a digit ' +
          'for each of &#9679; &#9670; &#9650;, check every clue, and backtrack on a contradiction. That is search. ' +
          'There is no left-to-right rule, so there is no forward chain to write down.</span>';
        state.textContent = "cipher hidden: only search works";
        btn.textContent = "Reveal the cipher key";
        if (ans) ans.textContent = "?";
      }
    }
    btn.addEventListener("click", function () { open = !open; draw(); });
    draw();
  })();

  /* ---------------- answer-key intervention ---------------- */
  (function () {
    var box = $("#cheat-btns"); if (!box) return;
    var CHEAT = {
      blind: { v: 0.03, note: "It has to crack the hidden code itself, which is pure searching. It almost never gets there." },
      half: { v: 0.05, note: "Half the code is handed over, but it still has to search for the rest, and that residue is enough to keep it stuck." },
      full: { v: 0.57, note: "The whole code is given. Now there is nothing to search: it reads the digits off and computes. Accuracy leaps." }
    };
    var bar = $("#cheat-bar"), num = $("#cheat-num"), note = $("#cheat-note");
    var btns = box.querySelectorAll(".cheat-btn");
    function pick(k) {
      var d = CHEAT[k], hot = k === "full";
      btns.forEach(function (b) { b.classList.toggle("active", b.dataset.k === k); });
      bar.style.width = Math.min(100, d.v / 0.6 * 100) + "%";
      bar.style.background = hot ? "var(--teal)" : "var(--clay)";
      num.textContent = Math.round(d.v * 100) + "%";
      num.style.color = hot ? "var(--teal)" : "var(--clay)";
      note.textContent = d.note;
    }
    btns.forEach(function (b) { b.addEventListener("click", function () { pick(b.dataset.k); }); });
    pick("blind");
  })();

  /* ---------------- scroll reveal + count-up ---------------- */
  function countUp(node) {
    var raw = node.getAttribute("data-count"), target = parseFloat(raw), dec = raw.indexOf(".") >= 0, t0 = null;
    function step(t) {
      if (!t0) t0 = t; var k = Math.min(1, (t - t0) / 950), e = 1 - Math.pow(1 - k, 3);
      node.textContent = dec ? (target * e).toFixed(2) : Math.round(target * e);
      if (k < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (en) {
        if (!en.isIntersecting) return;
        en.target.classList.add("in");
        if (en.target.querySelectorAll) en.target.querySelectorAll("[data-count]").forEach(countUp);
        io.unobserve(en.target);
      });
    }, { threshold: 0.16 });
    document.querySelectorAll(".reveal").forEach(function (n) { io.observe(n); });
  } else {
    document.querySelectorAll(".reveal").forEach(function (n) { n.classList.add("in"); });
  }
  window.__revealed = true; // tells the inline failsafe the observer is live

  /* ---------------- signature-catalog demo (Section 9) ---------------- */
  (function () {
    var eq = $("#sig-eq"); if (!eq) return;
    var C = '<span class="g s1">&#9679;</span>', D = '<span class="g s2">&#9670;</span>', T = '<span class="g s3">&#9650;</span>';
    eq.innerHTML = '<div class="eq"><span class="lab">clue</span>' + D + C + ' + ' + C + D + ' = ' + T + T + '</div>' +
      '<div class="eq" id="sig-line"></div>';
    var work = $("#sig-work"), state = $("#sig-state"), btn = $("#sigBtn"), stage = 0;
    var stages = [
      { st: "step 1 / 3", bt: "Recall the catalog",
        line: '<span class="lab">signature</span><span class="sig">A B B A C C</span>',
        wk: '<span class="srow">read the repeated symbols:&nbsp; &#9670;=A, &#9679;=B, &#9650;=C</span>' +
            '<span class="srow">&#9670;&#9679; + &#9679;&#9670; = &#9650;&#9650; &nbsp;maps to signature&nbsp; <b>ABBACC</b></span>' +
            '<span class="srow mut">every puzzle with this shape shares one short candidate list</span>' },
      { st: "step 2 / 3", bt: "Cross-check and solve",
        wk: '<span class="skey">search from scratch:&nbsp; up to 5&times;10<sup>10</sup> assignments</span>' +
            '<span class="srow">the model <b>recalls</b> catalog[ABBACC]:&nbsp; 6 candidates, no search</span>' +
            '<span class="srow">&#9670;=2 &#9679;=3 &#9650;=5 &nbsp;(23 + 32 = 55)</span>' +
            '<span class="srow">&#9670;=1 &#9679;=4 &#9650;=5 &nbsp;(14 + 41 = 55)</span>' +
            '<span class="srow">&#9670;=2 &#9679;=6 &#9650;=8 &nbsp;(26 + 62 = 88)</span>' +
            '<span class="srow mut">... and 3 more (every &#9670; + &#9679; = &#9650;)</span>' },
      { st: "memorized, then verified", bt: "Reset",
        wk: '<span class="srow">the puzzle\'s other clues keep just one:&nbsp; <b>&#9670;=2 &#9679;=3 &#9650;=5</b></span>' +
            '<span class="srow">solve &#9650;&#9679; + &#9670;&#9670; = 53 + 22 = 75</span>' +
            '<span class="srow big">answer&nbsp; <span class="box">75</span></span>' }
    ];
    function draw() {
      var line = $("#sig-line");
      if (stage === 0) {
        line.innerHTML = ""; work.classList.remove("show");
        work.innerHTML = '<span class="mut">A search over every digit assignment is hopeless in 7,680 tokens. Here is how the winners route around it.</span>';
        state.textContent = ""; btn.innerHTML = "Find the signature"; return;
      }
      var s = stages[stage - 1];
      if (s.line !== undefined) line.innerHTML = s.line;
      work.classList.remove("show"); work.innerHTML = s.wk;
      state.textContent = s.st; btn.innerHTML = s.bt;
      requestAnimationFrame(function () {
        work.querySelectorAll(".srow").forEach(function (n, k) { n.style.transitionDelay = (0.05 + k * 0.1) + "s"; });
        work.classList.add("show");
      });
    }
    btn.addEventListener("click", function () { stage = (stage + 1) % 4; draw(); });
    draw();
  })();

  /* ---------------- reading-progress bar ---------------- */
  (function () {
    var bar = $("#progress"); if (!bar) return;
    function upd() { var h = document.documentElement, m = h.scrollHeight - h.clientHeight; bar.style.width = (m > 0 ? (h.scrollTop / m) * 100 : 0) + "%"; }
    addEventListener("scroll", upd, { passive: true }); addEventListener("resize", upd); upd();
  })();

  /* ---------------- section dot-nav ---------------- */
  (function () {
    var dn = $("#dotnav"); if (!dn) return;
    var secs = [["premise", "The premise"], ["tasks", "The nine tasks"], ["surprise", "The surprise"], ["demo", "Forward vs. search"], ["why", "The mechanism"], ["training", "It fit the data"], ["scale", "Not the model"], ["cheat", "The causal proof"], ["solved", "How it was solved"], ["meaning", "What it means"], ["paper", "The paper"]];
    var map = {};
    secs.forEach(function (s) {
      var sec = document.getElementById(s[0]); if (!sec) return;
      var a = el("a"); a.href = "#" + s[0]; a.setAttribute("data-label", s[1]); a.setAttribute("aria-label", s[1]);
      dn.appendChild(a); map[s[0]] = a;
    });
    if ("IntersectionObserver" in window) {
      var io2 = new IntersectionObserver(function (es) {
        es.forEach(function (en) {
          if (en.isIntersecting) Object.keys(map).forEach(function (k) { map[k].classList.toggle("on", k === en.target.id); });
        });
      }, { rootMargin: "-45% 0px -45% 0px" });
      secs.forEach(function (s) { var sec = document.getElementById(s[0]); if (sec) io2.observe(sec); });
    }
  })();

  /* ---------------- code copy buttons + heading anchors ---------------- */
  (function () {
    document.querySelectorAll("pre.code").forEach(function (pre) {
      var b = el("button", "copy-btn"); b.type = "button"; b.textContent = "copy";
      b.addEventListener("click", function () {
        var c = pre.cloneNode(true), bb = c.querySelector(".copy-btn"); if (bb) bb.remove();
        if (navigator.clipboard) navigator.clipboard.writeText(c.innerText.trim()).then(function () {
          b.textContent = "copied"; b.classList.add("done");
          setTimeout(function () { b.textContent = "copy"; b.classList.remove("done"); }, 1400);
        });
      });
      pre.appendChild(b);
    });
    document.querySelectorAll("section[id]").forEach(function (sec) {
      var h = sec.querySelector("h2.h2"); if (!h) return;
      var a = el("a", "anchor"); a.href = "#" + sec.id; a.textContent = "#"; a.setAttribute("aria-label", "link to this section");
      h.insertBefore(a, h.firstChild);
    });
  })();

  /* ---------------- nav ---------------- */
  var nav = $("#nav");
  if (nav) addEventListener("scroll", function () { nav.classList.toggle("solid", scrollY > 10); }, { passive: true });

  /* ---------------- failure-trace explorer ---------------- */
  var TR = window.TRACES || [], i = 0;
  function hl(line) {
    return esc(line).replace(/(none\s+match[a-z]*\s*-&gt;\s*drop|-&gt;\s*drop|none\s+match[a-z]*)/gi, '<span class="hit">$1</span>');
  }
  function dots() {
    var host = $("#dots"); if (!host) return; host.innerHTML = "";
    TR.forEach(function (_, k) {
      var d = el("span", "dot" + (k === i ? " on" : ""));
      d.addEventListener("click", function () { i = k; render(); });
      host.appendChild(d);
    });
  }
  function render() {
    if (!$("#tline")) return;
    if (!TR.length) { $("#tline").textContent = "No traces loaded."; return; }
    var card = $("#traceCard"); card.classList.add("fade");
    setTimeout(function () {
      var t = TR[i];
      $("#tid").textContent = t.id || ("case " + (i + 1));
      $("#ttype").textContent = (t.type || "verdict-as-token") + (t.family ? " / " + t.family : "");
      $("#tline").innerHTML = hl(t.line);
      $("#twhy").textContent = t.why;
      $("#counter").textContent = (i + 1) + " / " + TR.length;
      dots(); card.classList.remove("fade");
    }, 150);
  }
  function go(d) { i = (i + d + TR.length) % TR.length; render(); }
  if ($("#next")) {
    $("#next").addEventListener("click", function () { go(1); });
    $("#prev").addEventListener("click", function () { go(-1); });
    addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight") go(1); else if (e.key === "ArrowLeft") go(-1);
    });
    render();
  }
})();
