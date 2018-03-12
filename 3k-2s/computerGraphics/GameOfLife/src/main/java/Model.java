public class Model {
//    public int width;
//    public int height;
    public int[][] field;

    public Model(int height, int width){
        this.field = new int[height][width];
//        this.height = height;
//        this.width = width;
    }

    public double getImpact(int x, int y){
        double impact = 0;

        if(x % 2 == 0){
            if(y > 0){
                impact+= this.field[x][y-1] * Params.FST_IMPACT;
            }
            if(y+1 < Params.modelWidth-1){

                impact+= this.field[x][y+1] * Params.FST_IMPACT;
            }
            if(x > 0 && y < Params.modelWidth-1){
                impact+= this.field[x-1][y] * Params.FST_IMPACT;
            }
            if(x > 0 && y > 0) {
                impact += this.field[x-1][y-1] * Params.FST_IMPACT;
            }
            if(x+1 < Params.modelHeight && y < Params.modelWidth-1 ){
                impact+= this.field[x+1][y] * Params.FST_IMPACT;
            }
            if(x+1 < Params.modelHeight && y > 0){
                impact += this.field[x+1][y-1] * Params.FST_IMPACT;
            }
//---------------------------------------------------------------------
            if(x+2 < Params.modelHeight){
                impact+= this.field[x+2][y] * Params.SND_IMPACT;
            }
            if(x-1 > 0){
                impact+= this.field[x-2][y] * Params.SND_IMPACT;
            }
            if(x > 0 && y-1 > 0){
                impact+= this.field[x-1][y-2] * Params.SND_IMPACT;
            }
            if(x > 0 && y+1 < Params.modelWidth-1){
                impact+= this.field[x-1][y+1] * Params.SND_IMPACT;
            }
            if(x+1 < Params.modelHeight && y-1 > 0){
                impact+= this.field[x+1][y-2] * Params.SND_IMPACT;
            }
            if(x+1 < Params.modelHeight && y+1 < Params.modelWidth-1){
                impact+= this.field[x+1][y+1] * Params.SND_IMPACT;
            }
        }
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
        else{
            if(y > 0){
                impact+= this.field[x][y-1] * Params.FST_IMPACT;
            }
            if(y+1 < Params.modelWidth-1){
                impact+= this.field[x][y+1] * Params.FST_IMPACT;
            }
            if(x > 0 && y+1 < Params.modelWidth){
                impact+= this.field[x-1][y] * Params.FST_IMPACT;
                impact+= this.field[x-1][y+1] * Params.FST_IMPACT;
            }
            if(x+1 < Params.modelHeight && y+1 < Params.modelWidth){
                impact+= this.field[x+1][y] * Params.FST_IMPACT;
                impact+= this.field[x+1][y+1] * Params.FST_IMPACT;
            }
//---------------------------------------------------------------------
            if(x+2 < Params.modelHeight){
                impact+= this.field[x+2][y] * Params.SND_IMPACT;
            }
            if(x-1 > 0){
                impact+= this.field[x-2][y] * Params.SND_IMPACT;
            }
            if(x > 0 && y > 0){
                impact+= this.field[x-1][y-1] * Params.SND_IMPACT;
            }
            if(x > 0 && y+2 < Params.modelWidth){
                impact+= this.field[x-1][y+2] * Params.SND_IMPACT;
            }
            if(x+1 < Params.modelHeight && y > 0){
                impact+= this.field[x+1][y-1] * Params.SND_IMPACT;
            }
            if(x+1 < Params.modelHeight && y+2 < Params.modelWidth){
                impact+= this.field[x+1][y+2] * Params.SND_IMPACT;
            }
        }
        return impact;
    }

    public void refrash(){
        int[][] newField = new int[Params.modelHeight][Params.modelWidth];
        for (int i = 0; i < Params.modelHeight; i++) {
            for (int j = 0; j < Params.modelWidth; j++) {
                //пропускаем последние у коротких рядов
                if(j == Params.modelWidth-1 && i%2 == 1){
                    continue;
                }
//                System.out.println(i+" ij "+j);
                double imp = getImpact(i, j);
                if (imp >= Params.BIRTH_BEGIN && imp <= Params.BIRTH_END
                        && this.field[i][j] == 0) {
                    newField[i][j] = 1;
                }
                if (imp >= Params.LIVE_BEGIN && imp <= Params.LIVE_END
                        && this.field[i][j] == 1)
                    newField[i][j] = 1;
                if((imp < Params.LIVE_BEGIN || imp > Params.LIVE_END)
                        && this.field[i][j] == 1){
                    newField[i][j] = 0;
                }
            }
        }
        field = newField;
    }
}
