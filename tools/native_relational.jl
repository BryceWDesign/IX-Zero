# Pure prediction comparison using the exact original source files supplied by the owner.
# No act!, native guard bypass, or execution of an external action. This is not a
# full-runtime benchmark. The script was authored but not executed in the build environment.
length(ARGS)==3 || error("usage: julia native_relational.jl BLACKFOX_ROOT AGI_ROOT FIXTURE_TSV")
include(joinpath(abspath(ARGS[1]),"src","IXBlackFoxAGI.jl"))
include(joinpath(abspath(ARGS[2]),"src","IXAGI.jl"))
const B=IXBlackFoxAGI
const A=IXAGI
function facts(M, raw)
    M.Situation(M.Fact(String(parts[1]),String.(parts[2:end])...) for parts in
        (split(f,'/') for f in split(raw,';') if !isempty(f)))
end
function command(M, raw)
    parts=split(raw,'/'); M.Command(String(parts[1]),String.(parts[2:end])...)
end
function score!(counts,pred,truth)
    key=pred===nothing ? "abstained" : pred==truth ? "correct" : "incorrect"
    counts[key]+=1
end
function main()
    left=Dict{String,B.WorldMemory}();right=Dict{String,A.CounterHemisphere}()
    results=Dict{Tuple{String,String},Dict{String,Int}}()
    for line in Iterators.drop(eachline(ARGS[3]),1)
        id,phase,before,act,after=String.(split(line,'\t';keepempty=true))
        w=get!(() -> B.WorldMemory(),left,id); h=get!(() -> A.CounterHemisphere(),right,id)
        bs,ba,bt=facts(B,before),command(B,act),facts(B,after)
        ags,aga,agt=facts(A.BF,before),command(A.BF,act),facts(A.BF,after)
        if phase in ("train","audit")
            # Donors may learn from the audit labels, unlike IX-Zero's frozen audit gate.
            B.learn!(w,bs,ba,bt);A.counter_learn!(h,ags,aga,agt)
        elseif phase=="test"
            lp=B.imagine(w,bs,ba);rp=A.predict_counter(h,ags,aga)
            for (arm,pred,truth) in (("IX-BlackFox-AGI WorldMemory",lp,bt),
                    ("IX-AGI CounterHemisphere",rp===nothing ? nothing : rp.state,agt))
                counts=get!(() -> Dict("correct"=>0,"incorrect"=>0,"abstained"=>0),results,(id,arm))
                score!(counts,pred,truth)
            end
        else
            error("unknown fixture phase")
        end
    end
    println("case\tarm\tcorrect\tincorrect\tabstained")
    for key in sort!(collect(keys(results)))
        r=results[key];println(join((key[1],key[2],r["correct"],r["incorrect"],r["abstained"]),'\t'))
    end
end
main()
