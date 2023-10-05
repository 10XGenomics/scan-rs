use crate::{
    betainc,
    consts::{MACHEP, MAXLOG, MINLOG},
    gammaln, ndtri,
};

enum State {
    IHalve,
    NewT,
}

pub fn betaincinv(aa: f64, bb: f64, yy0: f64) -> f64 {
    use State::{IHalve, NewT};
    if yy0 <= 0.0 {
        return 0.0;
    }
    if yy0 >= 1.0 {
        return 1.0;
    }
    let mut a: f64;
    let mut b: f64;
    let mut x: f64;
    let mut y: f64;
    let mut y0: f64;
    let mut rflg: bool;
    let mut state: State;
    let mut dithresh: f64;

    let mut x0 = 0.0;
    let mut yl = 0.0;
    let mut x1 = 1.0;
    let mut yh = 1.0;

    if aa <= 1.0 || bb <= 1.0 {
        dithresh = 1e-6;
        rflg = false;
        a = aa;
        b = bb;
        y0 = yy0;
        x = a / (a + b);
        y = betainc(a, b, x);
        state = IHalve;
    } else {
        dithresh = 1e-4;

        let yp = if yy0 > 0.5 {
            rflg = true;
            a = bb;
            b = aa;
            y0 = 1.0 - yy0;
            ndtri(yy0)
        } else {
            rflg = false;
            a = aa;
            b = bb;
            y0 = yy0;
            -ndtri(yy0)
        };

        let lgm = (yp * yp - 3.0) / 6.0;
        x = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let d = yp * (x + lgm).sqrt() / x
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (lgm + 5.0 / 6.0 - 2.0 / (3.0 * x));
        let d = 2.0 * d;
        if d < MINLOG {
            // underflow, x <- 0.0
            if rflg {
                return 1.0 - MACHEP;
            } else {
                return 0.0;
            }
        } else {
            x = a / (a + b * d.exp());
            y = betainc(a, b, x);
            let yp = (y - y0) / y0;
            if yp.abs() < 0.2 {
                state = NewT;
            } else {
                state = IHalve;
            }
        }
    }

    let mut nflg = false;
    'outer: loop {
        match state {
            IHalve => {
                let mut dir = 0i32;
                let mut di = 0.5;
                for i in 0..100 {
                    if i != 0 {
                        x = x0 + di * (x1 - x0);
                        if x == 1.0 {
                            x = 1.0 - MACHEP;
                        }
                        if x == 0.0 {
                            di = 0.5;
                            x = x0 + di * (x1 - x0);
                            if x == 0.0 {
                                break 'outer;
                            }
                        }
                        y = betainc(a, b, x);
                        let yp = (x1 - x0) / (x1 + x0);
                        if yp.abs() < dithresh {
                            state = NewT;
                            continue 'outer;
                        }
                        let yp = (y - y0) / y0;
                        if yp.abs() < dithresh {
                            state = NewT;
                            continue 'outer;
                        }
                    }
                    if y < y0 {
                        x0 = x;
                        yl = y;
                        if dir < 0 {
                            dir = 0;
                            di = 0.5;
                        } else if dir > 3 {
                            di = 1.0 - (1.0 - di) * (1.0 - di);
                        } else if dir > 1 {
                            di = 0.5 * di + 0.5;
                        } else {
                            di = (y0 - y) / (yh - yl);
                        }
                        dir += 1;
                        if x0 > 0.75 {
                            if rflg {
                                rflg = false;
                                a = aa;
                                b = bb;
                                y0 = yy0;
                            } else {
                                rflg = true;
                                a = bb;
                                b = aa;
                                y0 = 1.0 - yy0;
                            }
                            x = 1.0 - x;
                            y = betainc(a, b, x);
                            x0 = 0.0;
                            yl = 0.0;
                            x1 = 1.0;
                            yh = 1.0;
                            // state already is IHalve
                            continue 'outer;
                        }
                    } else {
                        if rflg && x < MACHEP {
                            x = 0.0;
                            break 'outer;
                        }
                        x1 = x;
                        yh = y;
                        if dir > 0 {
                            dir = 0;
                            di = 0.5;
                        } else if dir < -3 {
                            di = di * di;
                        } else if dir < -1 {
                            di *= 0.5;
                        } else {
                            di = (y - y0) / (yh - yl);
                        }
                        dir -= 1;
                    }
                }
                if x0 >= 1.0 {
                    x = 1.0 - MACHEP;
                    break 'outer;
                }
                if x <= 0.0 {
                    x = 0.0;
                    break 'outer;
                }
                state = NewT;
            }
            NewT => {
                if nflg {
                    break 'outer;
                }
                nflg = true;
                let lgm = gammaln(a + b) - gammaln(a) - gammaln(b);

                for i in 0..8 {
                    if i != 0 {
                        y = betainc(a, b, x);
                    }
                    if y < yl {
                        x = x0;
                        y = yl;
                    } else if y > yh {
                        x = x1;
                        y = yh;
                    } else if y < y0 {
                        x0 = x;
                        yl = y;
                    } else {
                        x1 = x;
                        yh = y;
                    }
                    if x == 1.0 || x == 0.0 {
                        break;
                    }
                    let d = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() + lgm;
                    if d < MINLOG {
                        break 'outer;
                    }
                    if d > MAXLOG {
                        break;
                    }
                    let d = d.exp();
                    let d = (y - y0) / d;
                    let mut xt = x - d;
                    if xt <= x0 {
                        y = (x - x0) / (x1 - x0);
                        xt = x0 + 0.5 * y * (x - x0);
                        if xt <= 0.0 {
                            break;
                        }
                    }
                    if xt >= x1 {
                        y = (x1 - x) / (x1 - x0);
                        xt = x1 - 0.5 * y * (x1 - x);
                        if xt >= 1.0 {
                            break;
                        }
                    }
                    x = xt;
                    if (d / x).abs() < 128.0 * MACHEP {
                        break 'outer;
                    }
                }
                dithresh = 256.0 * MACHEP;
                state = IHalve;
                continue 'outer;
            }
        }
    }

    if rflg {
        if x < MACHEP {
            1.0 - MACHEP
        } else {
            1.0 - x
        }
    } else {
        x
    }
}
