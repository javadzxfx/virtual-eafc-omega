# main.py — UltraGOD OMEGA v4 (Kivy UI)
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window

from core.omega_engine import OmegaSingleMarket, append_log

Window.clearcolor=(0.06,0.07,0.09,1)

class Root(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=8, padding=12, **kwargs)
        self.engine = OmegaSingleMarket()

        self.add_widget(Label(text="[b]Virtual EAFC — UltraGOD OMEGA v4[/b]", markup=True, size_hint=(1,0.07)))

        self.sp = Spinner(text="Select Market", values=("1X2","OU"), size_hint=(1,0.07))
        self.add_widget(self.sp)

        # Odds
        row = BoxLayout(size_hint=(1,0.085), spacing=6)
        self.t1 = TextInput(hint_text="Odds 1 (Home or Over)", multiline=False, input_filter="float")
        self.t2 = TextInput(hint_text="Odds 2 (Draw or Under)", multiline=False, input_filter="float")
        self.t3 = TextInput(hint_text="Odds 3 (Away) — for 1X2", multiline=False, input_filter="float")
        row.add_widget(self.t1); row.add_widget(self.t2); row.add_widget(self.t3)
        self.add_widget(row)

        # OU lines
        lrow = BoxLayout(size_hint=(1,0.09), spacing=6)
        self.line1 = TextInput(hint_text="O/U line #1 (e.g., 14.5)", multiline=False, input_filter="float")
        self.line2 = TextInput(hint_text="O/U line #2 (optional)", multiline=False, input_filter="float")
        self.ou2_o = TextInput(hint_text="Over@line2", multiline=False, input_filter="float")
        self.ou2_u = TextInput(hint_text="Under@line2", multiline=False, input_filter="float")
        lrow.add_widget(Label(text="Lines:", size_hint=(0.2,1)))
        lrow.add_widget(self.line1); lrow.add_widget(self.line2); lrow.add_widget(self.ou2_o); lrow.add_widget(self.ou2_u)
        self.add_widget(lrow)

        # Risk controls
        rrow = BoxLayout(size_hint=(1,0.09), spacing=6)
        self.bank = TextInput(hint_text="Bankroll (optional)", multiline=False, input_filter="float")
        self.alpha = TextInput(hint_text="CVaR α (e.g., 0.10)", multiline=False, input_filter="float")
        self.ri_thr = TextInput(hint_text="RI threshold (e.g., 0.60)", multiline=False, input_filter="float")
        self.smargin = TextInput(hint_text="Safety margin EV_low (e.g., 0.01)", multiline=False, input_filter="float")
        rrow.add_widget(self.bank); rrow.add_widget(self.alpha); rrow.add_widget(self.ri_thr); rrow.add_widget(self.smargin)
        self.add_widget(rrow)

        # Buttons
        brow = BoxLayout(size_hint=(1,0.085), spacing=6)
        self.btn = Button(text="Analyze"); self.btn.bind(on_release=self.run)
        self.btn_log = Button(text="Analyze + Log"); self.btn_log.bind(on_release=self.run_and_log)
        brow.add_widget(self.btn); brow.add_widget(self.btn_log)
        self.add_widget(brow)

        # Output
        self.out = TextInput(readonly=True, font_name="Roboto", size_hint=(1,0.595))
        self.add_widget(self.out)

    def _f(self, w):
        try: return float(w.text) if w.text else None
        except: return None

    def _analyze(self):
        try:
            kind = self.sp.text.strip()
            if kind not in ("1X2","OU"):
                return False, "Choose market (1X2/OU).", None

            o1,o2,o3 = self._f(self.t1), self._f(self.t2), self._f(self.t3)
            bank = self._f(self.bank)
            alpha = self._f(self.alpha); ri_thr = self._f(self.ri_thr); sm = self._f(self.smargin)

            # apply controls if provided
            if alpha is not None: self.engine.alpha = alpha
            if ri_thr is not None: self.engine.ri_thr = ri_thr
            if sm is not None: self.engine.s_margin = sm

            if kind=="1X2":
                if (o1 or 0)<=1.0 or (o2 or 0)<=1.0 or (o3 or 0)<=1.0:
                    return False, "Enter three odds > 1.0 for 1X2.", None
                res = self.engine.analyze("1X2", (o1,o2,o3), bank=bank)
            else:
                L1 = self._f(self.line1)
                if (o1 or 0)<=1.0 or (o2 or 0)<=1.0 or L1 is None:
                    return False, "Enter OU odds > 1.0 and line #1.", None
                aux=None
                L2=self._f(self.line2); O2=self._f(self.ou2_o); U2=self._f(self.ou2_u)
                if L2 is not None and O2 and U2 and O2>1.0 and U2>1.0:
                    aux={"line2":L2, "odds2":(O2,U2)}
                res = self.engine.analyze("OU", (o1,o2), line=L1, aux_ou=aux, bank=bank)

            lines=[]
            for r in res["rows"]:
                lines.append(
                    f"{r['name']:24s} | odds={r['odds']:.3f} | "
                    f"p∈[{r['p_low']:.4f},{r['p_high']:.4f}] | "
                    f"EV∈[{r['EV_low']:+.4f},{r['EV_high']:+.4f}] | "
                    f"Med={r['EV_med']:+.4f} | CVaR={r['EV_cvar']:+.4f} | RI={r['RI']:.2f}"
                )
            if res["final"]:
                f=res["final"]; lines.append("\nFINAL PICK (minimax + CVaR-safe):")
                lines.append(f"  {f['name']} | odds={f['odds']:.3f} | EV_low={f['EV_low']:+.4f} | CVaR={f['EV_cvar']:+.4f} | RI={f['RI']:.2f}")
                if res["stake"]:
                    s=res["stake"]; lines.append(f"  Stake ≈ {100*s['fraction']:.2f}% (~{s['amount']:.2f}) | pα≈{s['p_alpha']:.4f}")
            else:
                lines.append("\nFINAL PICK: No signal (fails EV_low/CVaR/RI/SafetyMargin).")

            return True, "\n".join(lines), res

        except Exception as e:
            return False, f"Error: {type(e).__name__}: {e}", None

    def run(self, *_):
        ok, txt, _ = self._analyze()
        self.out.text = txt

    def run_and_log(self, *_):
        ok, txt, res = self._analyze()
        self.out.text = txt
        if ok and res:
            entry={"input_output":res}
            ok2, path = append_log(entry)
            self.out.text += ("\n\n[Logged at] "+path) if ok2 else ("\n\n[Log failed] "+path)

class UltraApp(App):
    def build(self): return Root()

if __name__=="__main__":
    UltraApp().run()
